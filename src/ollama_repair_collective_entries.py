"""Wykrywanie i naprawianie nierozdzielonych hasel zbiorczych SGKP.

Skrypt dziala dwuetapowo:

1. wyszukuje rekordy ``indywidualne`` zawierajace numerowane czesci, oznacza
   wszystkie wystapienia numeracji stabilnymi identyfikatorami i prosi model
   Ollama o wskazanie granic podhasel najwyzszego poziomu oraz przypisanie
   istniejacych danych do elementow;
2. po recznym zatwierdzeniu identyfikatorow zapisuje poprawiona kopie pliku
   JSON oraz zapewnia unikalnosc ID w obrebie tomu. Plik zrodlowy nigdy nie
   jest modyfikowany.

Model nie przepisuje wartosci istniejacych pol. Otrzymuje ponumerowane
jednostki danych i zwraca tylko ich przypisanie do elementow. Wlasciwe
wartosci sa kopiowane z rekordu zrodlowego.
"""

from __future__ import annotations

import argparse
import copy
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_MODEL = "gemma4:31b-cloud"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_OUTPUT_DIR = Path("sgkp_naprawa/ollama")
DEFAULT_WORKERS = 1
DEFAULT_TIMEOUT = 300.0
DEFAULT_RETRIES = 5
DEFAULT_RETRY_DELAY = 30.0
DEFAULT_SETTLEMENT_TYPES_PATH = (
    Path(__file__).resolve().parents[2]
    / "sgkp_information_extraction"
    / "dictionary"
    / "typy_punktow_osadniczych_v2.csv"
)
DEFAULT_ABBREVIATIONS_PATH = (
    Path(__file__).resolve().parents[2]
    / "sgkp_information_extraction"
    / "dictionary"
    / "prompt_sgkp_skroty.txt"
)
PROMPT_VERSION = "collective_repair_ollama_v4"
TRANSIENT_HTTP_CODES = {408, 425, 429, 500, 502, 503, 504}

STRUCTURAL_PARENT_FIELDS = {
    "nazwa",
    "text",
    "tom",
    "strona",
    "rodzaj",
    "elementy",
    "ID",
    "autor",
}
REGENERATED_ELEMENT_FIELDS = {
    "typ",
    "typ_punktu_osadniczego",
    "opis_lokalizacji",
}

# Rozpoznaje m.in. ``1)``, ``1.)``, ``2 )``. Nie rozpoznaje zwyklych dat
# ani liczb zakonczonych kropka, co ogranicza liczbe falszywych kandydatow.
NUMBERED_MARKER_RE = re.compile(r"(?<!\d)(\d{1,3})\s*(?:\.\s*)?\)")


class OllamaTransientError(RuntimeError):
    """Blad tymczasowy, po ktorym warto ponowic zapytanie."""


class ModelResponseError(ValueError):
    """Niepoprawna odpowiedz modelu wraz z jej surowa trescia."""

    def __init__(self, message: str, raw_output: str = "") -> None:
        super().__init__(message)
        self.raw_output = raw_output


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.startswith("export "):
                line = line.removeprefix("export ").strip()
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if not key or key in os.environ:
                continue
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]
            os.environ[key] = value


def load_env_files(input_path: Path) -> None:
    candidates = [
        Path.cwd() / ".env",
        Path(__file__).resolve().parent / ".env",
        input_path.resolve().parent / ".env",
    ]
    seen: set[Path] = set()
    for path in candidates:
        if path not in seen:
            seen.add(path)
            load_env_file(path)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def record_hash(record: dict[str, Any]) -> str:
    return sha256_text(canonical_json(record))


def get_record_id(record: dict[str, Any]) -> str:
    return str(record.get("ID", "") or "").strip()


def record_label(record: dict[str, Any]) -> str:
    record_id = get_record_id(record)
    name = " ".join(str(record.get("nazwa", "") or "").split())
    return f"{record_id} {name}".strip()


def format_duration(seconds: float) -> str:
    total_seconds = int(round(seconds))
    minutes, secs = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def short_error(error: Exception) -> str:
    message = " ".join((str(error).strip() or error.__class__.__name__).split())
    return message if len(message) <= 500 else message[:497] + "..."


def iter_individual_records(data: Any) -> Iterator[dict[str, Any]]:
    records = data if isinstance(data, list) else [data]
    for record in records:
        if isinstance(record, dict) and record.get("rodzaj") == "indywidualne":
            yield record


def sequential_markers(text: str, name: str) -> list[re.Match[str]]:
    """Zwraca najdluzsza sekwencje znacznikow 1, 2, 3... blisko naglowka."""

    matches = list(NUMBERED_MARKER_RE.finditer(text))
    best: list[re.Match[str]] = []
    max_first_position = max(250, len(name) + 100)
    for index, match in enumerate(matches):
        if int(match.group(1)) != 1 or match.start() > max_first_position:
            continue
        sequence = [match]
        expected = 2
        for following in matches[index + 1 :]:
            number = int(following.group(1))
            if number == expected:
                sequence.append(following)
                expected += 1
            elif number == 1:
                break
        if len(sequence) > len(best):
            best = sequence
    return best if len(best) >= 2 else []


def split_numbered_text(
    text: str,
    markers: list[re.Match[str]],
) -> list[dict[str, Any]]:
    """Dzieli tekst po poczatkach numerowanych czesci bez zmiany znakow."""

    segments: list[dict[str, Any]] = []
    for index, marker in enumerate(markers):
        start = 0 if index == 0 else marker.start()
        end = markers[index + 1].start() if index + 1 < len(markers) else len(text)
        segment_text = text[start:end].strip()
        segments.append(
            {
                "segment_index": index + 1,
                "nr": str(marker.group(1)),
                "source_start": start,
                "source_end": end,
                "text": segment_text,
            }
        )
    return segments


def numbered_marker_candidates(text: str) -> list[dict[str, Any]]:
    """Nadaje kazdemu znacznikowi numeracji stabilny identyfikator.

    Zachowujemy także znaczniki z powtorzonymi i nieciaglymi numerami. Model
    wybiera sposrod nich granice podhasel redakcyjnych, zamiast dostawac z gory
    narzucony (i czasem bledny) podzial 1, 2, 3...
    """

    return [
        {
            "marker_id": f"m{index:03d}",
            "nr": match.group(1),
            "source_start": match.start(),
            "source_end": match.end(),
            "source_text": match.group(0),
        }
        for index, match in enumerate(NUMBERED_MARKER_RE.finditer(text), start=1)
    ]


def annotate_numbered_markers(
    text: str,
    markers: list[dict[str, Any]],
) -> str:
    """Wstawia techniczne etykiety przed znacznikami bez zmiany zrodla."""

    chunks: list[str] = []
    position = 0
    for marker in markers:
        start = int(marker["source_start"])
        chunks.append(text[position:start])
        chunks.append(f"[[{marker['marker_id']}]]")
        position = start
    chunks.append(text[position:])
    return "".join(chunks)


def segments_from_selected_markers(
    text: str,
    markers: list[dict[str, Any]],
    elements: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Buduje dokladne fragmenty tekstu z granic wskazanych przez model."""

    marker_by_id = {item["marker_id"]: item for item in markers}
    segments: list[dict[str, Any]] = []
    for position, element in enumerate(elements):
        marker_id = element["start_marker_id"]
        if position == 0:
            start = 0
        else:
            start = int(marker_by_id[marker_id]["source_start"])
        if position + 1 < len(elements):
            next_marker_id = elements[position + 1]["start_marker_id"]
            end = int(marker_by_id[next_marker_id]["source_start"])
        else:
            end = len(text)
        nr = "1" if marker_id == "START" else str(marker_by_id[marker_id]["nr"])
        segments.append(
            {
                "segment_index": int(element["segment_index"]),
                "nr": nr,
                "start_marker_id": marker_id,
                "source_start": start,
                "source_end": end,
                "text": text[start:end].strip(),
            }
        )
    return segments


def find_candidates(data: Any) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for record in iter_individual_records(data):
        text = str(record.get("text", "") or "")
        name = str(record.get("nazwa", "") or "")
        markers = sequential_markers(text, name)
        if not markers:
            continue
        segments = split_numbered_text(text, markers)
        if any(len(segment["text"]) < 15 for segment in segments):
            continue
        candidates.append(
            {
                "record": record,
                "segments": segments,
                "marker_candidates": numbered_marker_candidates(text),
                "reason": f"sekwencja numerowana 1-{len(segments)}",
            }
        )
    return candidates


def build_data_units(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Rozbija dane rekordu na jednostki, ktore model moze tylko przypisac."""

    units: list[dict[str, Any]] = []
    unit_number = 1
    excluded = STRUCTURAL_PARENT_FIELDS | REGENERATED_ELEMENT_FIELDS
    for field, value in record.items():
        if field in excluded:
            continue
        if isinstance(value, list):
            for item_index, item in enumerate(value):
                units.append(
                    {
                        "unit_id": f"u{unit_number:04d}",
                        "field": field,
                        "kind": "list_item",
                        "item_index": item_index,
                        "value": item,
                    }
                )
                unit_number += 1
        else:
            units.append(
                {
                    "unit_id": f"u{unit_number:04d}",
                    "field": field,
                    "kind": "whole_field",
                    "value": value,
                }
            )
            unit_number += 1
    return units


def load_settlement_type_mapping(path: Path) -> dict[str, str]:
    """Wczytuje eksperckie mapowanie typu hasla na typ punktu osadniczego."""

    mapping: dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"typ_model", "typ_punktu_osadniczego"}
        if not reader.fieldnames or not required <= set(reader.fieldnames):
            raise RuntimeError(f"Niepoprawny plik typow punktow osadniczych: {path}")
        for row in reader:
            source = str(row.get("typ_model", "") or "").strip()
            target = str(row.get("typ_punktu_osadniczego", "") or "").strip()
            if source and target:
                mapping[source.casefold()] = target
    return mapping


def load_abbreviations(path: Path) -> tuple[str, dict[str, str]]:
    """Wczytuje tekst promptu oraz mapowanie skrot -> rozwiniecie."""

    lines: list[str] = []
    mapping: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            lines.append(line)
            if "=" not in line:
                continue
            abbreviation, expansion = line.split("=", 1)
            abbreviation = abbreviation.strip()
            expansion = expansion.strip()
            if abbreviation and expansion:
                mapping[abbreviation.casefold()] = expansion
    return "\n".join(lines), mapping


def normalize_entry_types(
    entry_types: list[str],
    abbreviation_mapping: dict[str, str],
) -> list[str]:
    """Rozwija typy zwrocone przez model w formie skrotow SGKP."""

    normalized: list[str] = []
    for entry_type in entry_types:
        value = entry_type.strip()
        expanded = abbreviation_mapping.get(value.casefold(), value).strip()
        if expanded and expanded not in normalized:
            normalized.append(expanded)
    return normalized


def settlement_types_for(
    entry_types: list[str],
    mapping: dict[str, str],
) -> tuple[list[str], list[str]]:
    settlement_types: list[str] = []
    unknown: list[str] = []
    for entry_type in entry_types:
        mapped = mapping.get(entry_type.strip().casefold())
        if mapped is None:
            unknown.append(entry_type)
        elif mapped.casefold() != "nie dotyczy" and mapped not in settlement_types:
            settlement_types.append(mapped)
    return settlement_types, unknown


def ollama_chat_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/api/chat"):
        return normalized
    if normalized.endswith("/api"):
        return normalized + "/chat"
    return normalized + "/api/chat"


def ollama_chat(
    base_url: str,
    api_key: str | None,
    model: str,
    prompt: str,
    timeout: float,
) -> str:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Odpowiadasz wylacznie poprawnym JSON-em. Nie dodawaj "
                    "markdown ani tekstu poza obiektem JSON."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {"temperature": 0},
    }
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = Request(
        ollama_chat_url(base_url),
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            response_text = response.read().decode("utf-8", errors="replace")
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        message = f"Ollama zwrocila HTTP {exc.code}: {body}"
        if exc.code in TRANSIENT_HTTP_CODES:
            raise OllamaTransientError(message) from exc
        raise RuntimeError(message) from exc
    except (URLError, TimeoutError) as exc:
        raise OllamaTransientError(f"Nie mozna polaczyc sie z Ollama: {base_url}") from exc

    try:
        response_data = json.loads(response_text)
    except json.JSONDecodeError as exc:
        raise OllamaTransientError("Ollama zwrocila niepoprawny JSON HTTP") from exc
    if not isinstance(response_data, dict):
        raise OllamaTransientError("Odpowiedz Ollamy nie jest obiektem")
    if response_data.get("error"):
        raise RuntimeError(f"Ollama zwrocila blad: {response_data['error']}")
    message = response_data.get("message")
    content = message.get("content", "") if isinstance(message, dict) else ""
    if not isinstance(content, str) or not content.strip():
        raise OllamaTransientError("Ollama zwrocila pusta odpowiedz")
    return content.strip()


def parse_model_json(output: str) -> dict[str, Any]:
    stripped = output.strip()
    candidates = [stripped]
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end >= start:
        candidates.append(stripped[start : end + 1])
    for candidate in list(candidates):
        if candidate.startswith("{{"):
            candidates.append(candidate[1:])
    errors: list[str] = []
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            errors.append(str(exc))
            continue
        if isinstance(parsed, dict):
            return parsed
        errors.append("element glowny nie jest obiektem")
    detail = errors[-1] if errors else "brak obiektu"
    raise ModelResponseError(f"Nie mozna odczytac JSON modelu: {detail}", output)


def build_prompt(
    record: dict[str, Any],
    marker_candidates: list[dict[str, Any]],
    units: list[dict[str, Any]],
    abbreviations_text: str,
) -> str:
    text = str(record.get("text", "") or "")
    marker_payload = [
        {
            "marker_id": item["marker_id"],
            "nr": item["nr"],
            "source_text": item["source_text"],
        }
        for item in marker_candidates
    ]
    annotated_text = annotate_numbered_markers(text, marker_candidates)
    return f"""Ocen, czy rekord oznaczony jako indywidualny zawiera kilka odrebnych PODHASEL REDAKCYJNYCH SGKP i powinien zostac rekordem zbiorczym.

Najwazniejsza jest struktura redakcyjna tekstu, a nie sama liczba opisanych nazw lub obiektow.

Wybierz "zbiorcze" tylko wtedy, gdy pod wspolnym naglowkiem umieszczono co najmniej dwa odrebne hasla/slownikowe znaczenia, zwykle rozdzielone numerami 1), 2), 3). Kazde takie podhaslo opisuje osobna pozycje SGKP.

Wybierz "indywidualne", gdy jeden wpis o wsi, gminie, dominium, dobrach, parafii itp. tylko WYLICZA nalezace do niego wsie, folwarki, osady, czesci albo inne skladniki. Zwroty takie jak "obejmuje N miejscowosci", "N miejsc.", "sklada sie z" oraz numerowana lista skladnikow wewnatrz opisu NIE tworza podhasel zbiorczych.

Przyklad pozytywny: "Babice, 1) wś ... 2) B., wś rządowa ... 3) B., wś ..." to haslo zbiorcze, bo kolejne numery rozpoczynaja niezalezne opisy miejsc o tej samej nazwie.

Przyklad negatywny: "Białczewin, gmina ... 3 miejscowości: 1) Białczewin; 2) leśnictwo; 3) folwark" pozostaje haslem indywidualnym, bo numery sa tylko wykazem skladnikow jednej gminy. Analogicznie wykaz miejscowosci jednego dominium nie jest haslem zbiorczym.

W tekscie przed kazdym wykrytym znacznikiem numeracji wstawiono techniczna etykiete [[mNNN]]. Etykiet nie ma w oryginalnym tekscie. Jezeli decyzja to "zbiorcze", wybierz tylko etykiety rozpoczynajace podhasla najwyzszego poziomu. Pomin numery list wewnetrznych, przypisy, daty i wyliczenia skladnikow. Pierwszy element moze miec start_marker_id="START", jesli zaczyna sie od poczatku tekstu bez wlasnego numeru. Powtorzone lub nieciagle numery sa mozliwe; identyfikator mNNN jednoznacznie wskazuje konkretne wystapienie.

Jesli decyzja to "zbiorcze":
- zwroc jeden element dla kazdego podhasla najwyzszego poziomu;
- zachowaj kolejnosc w tekscie i nadaj element_index kolejno 1, 2, 3...;
- kazde start_marker_id moze wystapic najwyzej raz;
- nie dziel pojedynczego podhasla na jego skladniki, nawet jesli wymienia kilka nazw geograficznych;
- podaj pelna nazwe obiektu, rozwijaj skroty typu "B." na podstawie nazwy hasla;
- pole typ ma zawierac pelne, rozwiniete nazwy typow, np. "wieś", "folwark", "osada", "miasto", "miasteczko", "rzeka";
- nigdy nie zapisuj w polu typ skrotow OCR takich jak "wś", "mko", "mczko", "folw.", "os." lub "rz.". Rozwin je zgodnie z ponizszym slownikiem;
- zwroc tylko pole typ; typ_punktu_osadniczego zostanie wyliczony osobno z katalogu eksperckiego;
- przypisz kazda JEDNOSTKE DANYCH do jednego lub wielu elementow albo umiesc ja w nieprzypisane_jednostki;
- opieraj przypisanie na tresci odpowiedniego podhasla. Nie zmieniaj i nie przepisuj wartosci jednostek;
- lokalizacje geograficzna przypisz tylko wtedy, gdy jednoznacznie dotyczy danego segmentu;
- sklejone pole opis_lokalizacji zostanie wygenerowane ponownie i nie jest jednostka do przypisania;
- nie tworz nowych danych administracyjnych ani statystycznych.

SLOWNIK SKROTOW SGKP:
{abbreviations_text}

Nazwa rekordu: {record.get('nazwa')}
ID rekordu: {get_record_id(record)}

ORYGINALNY TEKST Z TECHNICZNYMI ETYKIETAMI ZNACZNIKOW:
{annotated_text}

KANDYDACI NA GRANICE PODHASEL:
{json.dumps(marker_payload, ensure_ascii=False, indent=2)}

JEDNOSTKI DANYCH Z ISTNIEJACEGO REKORDU:
{json.dumps(units, ensure_ascii=False, indent=2)}

Zwroc wylacznie obiekt JSON:
{{
  "decyzja": "zbiorcze",
  "pewnosc": "wysoka",
  "uzasadnienie": "krotkie uzasadnienie",
  "elementy": [
    {{
      "element_index": 1,
      "start_marker_id": "m001",
      "nazwa": "pelna nazwa",
      "typ": ["wieś", "folwark"]
    }},
    {{
      "element_index": 2,
      "start_marker_id": "m002",
      "nazwa": "pelna nazwa",
      "typ": ["wieś"]
    }}
  ],
  "przypisania": [
    {{"unit_id": "u0001", "element_indexes": [1]}}
  ],
  "nieprzypisane_jednostki": [
    {{"unit_id": "u0002", "powod": "dane laczne lub niejednoznaczne"}}
  ],
  "ostrzezenia": []
}}

Dozwolone decyzje: "zbiorcze", "indywidualne", "niepewne".
Dozwolona pewnosc: "wysoka", "srednia", "niska".
Dla decyzji "zbiorcze" musza istniec co najmniej dwa elementy, a kazdy unit_id musi wystapic dokladnie raz: albo w przypisaniach, albo w nieprzypisane_jednostki. Dla decyzji "indywidualne" lub "niepewne" elementy i przypisania maja byc puste."""


def string_list(value: Any, field: str) -> list[str]:
    """Normalizuje liste napisow, tolerujac null i pojedynczy napis modelu."""

    if value is None:
        return []
    if isinstance(value, str):
        normalized = value.strip()
        if normalized.casefold() in {
            "",
            "brak",
            "null",
            "none",
            "nie dotyczy",
        }:
            return []
        return [normalized]
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"Pole {field} nie jest lista napisow")
    return [item.strip() for item in value if item.strip()]


def warning_list(value: Any) -> list[str]:
    """Tolerancyjnie normalizuje techniczne ostrzezenia modelu."""

    if value is None:
        return []
    items = value if isinstance(value, list) else [value]
    result: list[str] = []
    for item in items:
        if isinstance(item, str):
            normalized = item.strip()
        else:
            normalized = canonical_json(item)
        if normalized and normalized.casefold() not in {"brak", "null", "none"}:
            result.append(normalized)
    return result


def normalize_zero_based_indexes(
    indexes: list[int],
    element_count: int,
) -> list[int]:
    """Toleruje czesty blad modelu: indeksy 0..n-1 zamiast 1..n."""

    if indexes and 0 in indexes and all(0 <= item < element_count for item in indexes):
        return [item + 1 for item in indexes]
    return indexes


def normalize_proposal(
    parsed: dict[str, Any],
    marker_candidates: list[dict[str, Any]],
    units: list[dict[str, Any]],
) -> dict[str, Any]:
    decision = str(parsed.get("decyzja", "") or "").strip()
    confidence = str(parsed.get("pewnosc", "") or "").strip()
    if decision not in {"zbiorcze", "indywidualne", "niepewne"}:
        raise ValueError(f"Nieznana decyzja: {decision!r}")
    if confidence not in {"wysoka", "srednia", "niska"}:
        raise ValueError(f"Nieznana pewnosc: {confidence!r}")

    if decision != "zbiorcze":
        return {
            "decyzja": decision,
            "pewnosc": confidence,
            "uzasadnienie": str(parsed.get("uzasadnienie", "") or "").strip(),
            "elementy": [],
            "przypisania": [],
            "nieprzypisane_jednostki": [],
            "ostrzezenia": warning_list(parsed.get("ostrzezenia", [])),
        }

    marker_by_id = {item["marker_id"]: item for item in marker_candidates}
    raw_elements = parsed.get("elementy")
    if not isinstance(raw_elements, list):
        raise ValueError("Pole elementy nie jest lista")
    if len(raw_elements) < 2:
        raise ValueError("Haslo zbiorcze musi miec co najmniej dwa elementy")

    prepared_elements: list[tuple[int, dict[str, Any]]] = []
    for raw in raw_elements:
        if not isinstance(raw, dict):
            raise ValueError("Element propozycji nie jest obiektem")
        try:
            index = int(raw.get("element_index", raw.get("segment_index")))
        except (TypeError, ValueError) as exc:
            raise ValueError("Niepoprawny element_index") from exc
        prepared_elements.append((index, raw))

    raw_indexes = [index for index, _ in prepared_elements]
    normalized_element_indexes = normalize_zero_based_indexes(
        raw_indexes, len(prepared_elements)
    )
    elements: dict[int, dict[str, Any]] = {}
    used_marker_ids: set[str] = set()
    previous_start = -1
    for index, (_, raw) in zip(normalized_element_indexes, prepared_elements):
        if index in elements:
            raise ValueError(f"Powtorzony element_index {index}")
        name = str(raw.get("nazwa", "") or "").strip()
        if not name:
            raise ValueError(f"Brak nazwy elementu {index}")
        marker_id = str(raw.get("start_marker_id", "") or "").strip()
        if index == 1 and marker_id == "START":
            marker_start = 0
        elif marker_id in marker_by_id:
            marker_start = int(marker_by_id[marker_id]["source_start"])
        else:
            raise ValueError(f"Nieznany start_marker_id dla elementu {index}")
        if marker_id in used_marker_ids:
            raise ValueError(f"Powtorzony start_marker_id {marker_id}")
        if marker_start <= previous_start:
            raise ValueError("Granice elementow nie sa w kolejnosci tekstu")
        used_marker_ids.add(marker_id)
        previous_start = marker_start
        elements[index] = {
            "segment_index": index,
            "start_marker_id": marker_id,
            "nazwa": name,
            "typ": string_list(raw.get("typ", []), "typ"),
        }
    element_indexes = set(range(1, len(elements) + 1))
    if set(elements) != element_indexes:
        raise ValueError("element_index musi tworzyc ciag 1, 2, 3...")

    unit_ids = {item["unit_id"] for item in units}
    raw_assignments = parsed.get("przypisania", [])
    raw_unassigned = parsed.get("nieprzypisane_jednostki", [])
    if not isinstance(raw_assignments, list) or not isinstance(raw_unassigned, list):
        raise ValueError("Przypisania lub nieprzypisane_jednostki nie sa lista")

    all_assignment_indexes: list[int] = []
    for raw in raw_assignments:
        if not isinstance(raw, dict) or not isinstance(raw.get("element_indexes"), list):
            continue
        try:
            all_assignment_indexes.extend(int(item) for item in raw["element_indexes"])
        except (TypeError, ValueError):
            pass
    assignments_are_zero_based = (
        0 in all_assignment_indexes
        and all(0 <= item < len(elements) for item in all_assignment_indexes)
    )

    raw_unassigned_ids = {
        str(item.get("unit_id", "") or "").strip()
        for item in raw_unassigned
        if isinstance(item, dict)
    }
    assignments: list[dict[str, Any]] = []
    seen_units: set[str] = set()
    for raw in raw_assignments:
        if not isinstance(raw, dict):
            raise ValueError("Przypisanie nie jest obiektem")
        unit_id = str(raw.get("unit_id", "") or "").strip()
        indexes = raw.get("element_indexes")
        if unit_id not in unit_ids or unit_id in seen_units:
            raise ValueError(f"Nieznana lub powtorzona jednostka {unit_id!r}")
        if isinstance(indexes, list) and not indexes and unit_id in raw_unassigned_ids:
            # Model czasem umieszcza jednostke jednoczesnie jako puste
            # przypisanie i jako nieprzypisana. Drugi zapis jest jednoznaczny.
            continue
        if not isinstance(indexes, list) or not indexes:
            raise ValueError(f"Brak element_indexes dla {unit_id}")
        try:
            normalized_indexes = [int(item) for item in indexes]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Niepoprawny indeks elementu dla {unit_id}") from exc
        if assignments_are_zero_based:
            normalized_indexes = [item + 1 for item in normalized_indexes]
        if any(item not in element_indexes for item in normalized_indexes):
            raise ValueError(f"Niepoprawny indeks elementu dla {unit_id}")
        if len(set(normalized_indexes)) != len(normalized_indexes):
            raise ValueError(f"Powtorzony indeks elementu dla {unit_id}")
        assignments.append(
            {"unit_id": unit_id, "element_indexes": normalized_indexes}
        )
        seen_units.add(unit_id)

    unassigned: list[dict[str, str]] = []
    for raw in raw_unassigned:
        if not isinstance(raw, dict):
            raise ValueError("Nieprzypisana jednostka nie jest obiektem")
        unit_id = str(raw.get("unit_id", "") or "").strip()
        if unit_id not in unit_ids or unit_id in seen_units:
            raise ValueError(f"Nieznana lub powtorzona jednostka {unit_id!r}")
        unassigned.append(
            {
                "unit_id": unit_id,
                "powod": str(raw.get("powod", "") or "").strip(),
            }
        )
        seen_units.add(unit_id)
    if seen_units != unit_ids:
        missing = sorted(unit_ids - seen_units)
        raise ValueError(f"Model nie rozliczyl jednostek: {missing}")

    return {
        "decyzja": decision,
        "pewnosc": confidence,
        "uzasadnienie": str(parsed.get("uzasadnienie", "") or "").strip(),
        "elementy": [elements[index] for index in sorted(elements)],
        "przypisania": assignments,
        "nieprzypisane_jednostki": unassigned,
        "ostrzezenia": warning_list(parsed.get("ostrzezenia", [])),
    }


def verify_with_retries(
    base_url: str,
    api_key: str | None,
    model: str,
    prompt: str,
    marker_candidates: list[dict[str, Any]],
    units: list[dict[str, Any]],
    label: str,
    timeout: float,
    retries: int,
    retry_delay: float,
) -> dict[str, Any]:
    attempts = max(1, retries)
    current_prompt = prompt
    for attempt in range(1, attempts + 1):
        output = ""
        try:
            output = ollama_chat(base_url, api_key, model, current_prompt, timeout)
            return normalize_proposal(
                parse_model_json(output), marker_candidates, units
            )
        except OllamaTransientError as exc:
            if attempt == attempts:
                raise
            print(
                f"{label}: blad tymczasowy ({attempt}/{attempts}): "
                f"{short_error(exc)}",
                file=sys.stderr,
            )
            time.sleep(max(0.0, retry_delay))
        except ValueError as exc:
            if attempt == attempts:
                raw_output = getattr(exc, "raw_output", output)
                raise ModelResponseError(short_error(exc), raw_output) from exc
            print(
                f"{label}: bledny wynik modelu ({attempt}/{attempts}): "
                f"{short_error(exc)}",
                file=sys.stderr,
            )
            current_prompt = (
                prompt
                + "\n\nPOPRZEDNIA ODPOWIEDZ BYLA NIEPOPRAWNA. "
                + f"Walidator zglosil: {short_error(exc)}. "
                + "Zwroc caly obiekt JSON ponownie, po poprawieniu tego bledu. "
                + "Nie dodawaj komentarza poza JSON.\n"
                + output
            )
            time.sleep(min(5.0, max(0.0, retry_delay)))
    raise RuntimeError("Nieudana ocena modelu")


def build_proposed_record(
    record: dict[str, Any],
    segments: list[dict[str, Any]],
    units: list[dict[str, Any]],
    proposal: dict[str, Any],
    settlement_type_mapping: dict[str, str],
) -> dict[str, Any] | None:
    if proposal["decyzja"] != "zbiorcze":
        return None

    unit_by_id = {item["unit_id"]: item for item in units}
    assigned: dict[int, list[dict[str, Any]]] = {
        item["segment_index"]: [] for item in segments
    }
    for assignment in proposal["przypisania"]:
        unit = unit_by_id[assignment["unit_id"]]
        for element_index in assignment["element_indexes"]:
            assigned[element_index].append(unit)

    metadata = {item["segment_index"]: item for item in proposal["elementy"]}
    parent_id = get_record_id(record)
    elements: list[dict[str, Any]] = []
    for segment in segments:
        index = segment["segment_index"]
        meta = metadata[index]
        element: dict[str, Any] = {
            "nazwa": meta["nazwa"],
            "nr": segment["nr"],
            "text": segment["text"],
            "rodzaj": "element",
            "ID": f"{parent_id}-{index:03d}",
            "typ": meta["typ"],
        }
        settlement_types, _ = settlement_types_for(
            meta["typ"], settlement_type_mapping
        )
        if settlement_types:
            element["typ_punktu_osadniczego"] = settlement_types

        list_fields: dict[str, list[tuple[int, Any]]] = {}
        for unit in assigned[index]:
            field = unit["field"]
            if unit["kind"] == "list_item":
                list_fields.setdefault(field, []).append(
                    (int(unit["item_index"]), unit["value"])
                )
            else:
                element[field] = unit["value"]
        for field, items in list_fields.items():
            element[field] = [value for _, value in sorted(items)]

        if "typ_punktu_osadniczego" in element:
            element["opis_lokalizacji"] = ""
        elements.append(element)

    parent: dict[str, Any] = {
        "nazwa": record.get("nazwa"),
        "text": record.get("text"),
    }
    for field in ("tom", "strona"):
        if field in record:
            parent[field] = record[field]
    parent["rodzaj"] = "zbiorcze"
    parent["elementy"] = elements
    parent["ID"] = parent_id
    if "autor" in record:
        parent["autor"] = record["autor"]
    return parent


def validation_key(source_name: str, record: dict[str, Any], model: str) -> str:
    payload = {
        "source": source_name,
        "record": record,
        "model": model,
        "prompt_version": PROMPT_VERSION,
    }
    return sha256_text(canonical_json(payload))


def proposal_risks(
    candidate: dict[str, Any],
    proposal: dict[str, Any],
    segments: list[dict[str, Any]],
) -> list[str]:
    """Wylicza sygnaly wymagajace szczegolnej kontroli czlowieka."""

    if proposal["decyzja"] == "indywidualne":
        return []
    risks: list[str] = []
    if proposal["decyzja"] == "niepewne":
        risks.append("decyzja_niepewna")
    if proposal["pewnosc"] != "wysoka":
        risks.append("pewnosc_nizsza_niz_wysoka")
    if proposal.get("nieprzypisane_jednostki"):
        risks.append("nieprzypisane_jednostki")
    if proposal.get("ostrzezenia"):
        risks.append("ostrzezenia_modelu_lub_katalogu")
    if proposal["decyzja"] == "zbiorcze":
        heuristic_count = len(candidate.get("segments", []))
        if heuristic_count != len(segments):
            risks.append("podzial_inny_niz_wstepna_heurystyka")
        numbers = [int(item["nr"]) for item in segments]
        if numbers != list(range(1, len(numbers) + 1)):
            risks.append("nietypowa_numeracja_podhasel")
        if any(len(item["text"]) < 15 for item in segments):
            risks.append("bardzo_krotki_segment")
    return risks


def make_proposal_row(
    source_path: Path,
    candidate: dict[str, Any],
    base_url: str,
    api_key: str | None,
    model: str,
    timeout: float,
    retries: int,
    retry_delay: float,
    settlement_type_mapping: dict[str, str],
    abbreviation_mapping: dict[str, str],
    abbreviations_text: str,
) -> dict[str, Any]:
    record = candidate["record"]
    marker_candidates = candidate["marker_candidates"]
    units = build_data_units(record)
    prompt = build_prompt(record, marker_candidates, units, abbreviations_text)
    proposal = verify_with_retries(
        base_url,
        api_key,
        model,
        prompt,
        marker_candidates,
        units,
        record_label(record),
        timeout,
        retries,
        retry_delay,
    )
    segments = (
        segments_from_selected_markers(
            str(record.get("text", "") or ""),
            marker_candidates,
            proposal["elementy"],
        )
        if proposal["decyzja"] == "zbiorcze"
        else []
    )
    for element in proposal.get("elementy", []):
        element["typ"] = normalize_entry_types(
            element.get("typ", []), abbreviation_mapping
        )
    unknown_types = sorted(
        {
            unknown_type
            for element in proposal.get("elementy", [])
            for unknown_type in settlement_types_for(
                element.get("typ", []), settlement_type_mapping
            )[1]
        }
    )
    if unknown_types:
        proposal["ostrzezenia"].append(
            "Typy nieobecne w katalogu eksperckim: " + ", ".join(unknown_types)
        )
    risks = proposal_risks(candidate, proposal, segments)
    return {
        "plik_zrodlowy": source_path.name,
        "ID": get_record_id(record),
        "nazwa": record.get("nazwa"),
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "validation_key": validation_key(source_path.name, record, model),
        "record_sha256": record_hash(record),
        "powod_kandydatury": candidate["reason"],
        "kandydaci_na_granice": marker_candidates,
        "segmenty": segments,
        "jednostki_danych": units,
        **proposal,
        "ryzyka": risks,
        "wymaga_szczegolnej_kontroli": bool(risks),
        "proponowany_rekord": build_proposed_record(
            record, segments, units, proposal, settlement_type_mapping
        ),
        "zatwierdzone": False,
        "czas_utc": datetime.now(timezone.utc).isoformat(),
    }


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False))
        handle.write("\n")


def make_error_row(
    source_path: Path,
    record: dict[str, Any],
    model: str,
    error: Exception,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "plik_zrodlowy": source_path.name,
        "ID": get_record_id(record),
        "nazwa": record.get("nazwa"),
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "validation_key": validation_key(source_path.name, record, model),
        "blad": short_error(error),
        "czas_utc": datetime.now(timezone.utc).isoformat(),
    }
    raw_output = getattr(error, "raw_output", "")
    if raw_output:
        row["surowa_odpowiedz_modelu"] = raw_output
    return row


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Niepoprawny JSONL {path}:{line_number}") from exc
            if isinstance(row, dict):
                rows.append(row)
    return rows


def write_json(path: Path, data: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")
    temporary.replace(path)


def write_candidate_csv(path: Path, candidates: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "ID",
                "nazwa",
                "liczba_segmentow_heurystyki",
                "liczba_wszystkich_znacznikow",
                "powod",
            ],
        )
        writer.writeheader()
        for candidate in candidates:
            writer.writerow(
                {
                    "ID": get_record_id(candidate["record"]),
                    "nazwa": candidate["record"].get("nazwa"),
                    "liczba_segmentow_heurystyki": len(candidate["segments"]),
                    "liczba_wszystkich_znacznikow": len(
                        candidate["marker_candidates"]
                    ),
                    "powod": candidate["reason"],
                }
            )
    temporary.replace(path)


def write_proposal_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    columns = [
        "ID",
        "nazwa",
        "decyzja",
        "pewnosc",
        "liczba_elementow",
        "nieprzypisane_jednostki",
        "wymaga_szczegolnej_kontroli",
        "ryzyka",
        "uzasadnienie",
        "ostrzezenia",
    ]
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "ID": row.get("ID"),
                    "nazwa": row.get("nazwa"),
                    "decyzja": row.get("decyzja"),
                    "pewnosc": row.get("pewnosc"),
                    "liczba_elementow": len(row.get("elementy", [])),
                    "nieprzypisane_jednostki": len(
                        row.get("nieprzypisane_jednostki", [])
                    ),
                    "wymaga_szczegolnej_kontroli": row.get(
                        "wymaga_szczegolnej_kontroli", True
                    ),
                    "ryzyka": " | ".join(row.get("ryzyka", [])),
                    "uzasadnienie": row.get("uzasadnienie"),
                    "ostrzezenia": " | ".join(row.get("ostrzezenia", [])),
                }
            )
    temporary.replace(path)


def read_approved_ids(path: Path) -> set[str]:
    approved: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            value = line.split("#", 1)[0].strip()
            if value:
                approved.add(value)
    return approved


def iter_record_ids(data: Any) -> Iterator[tuple[str, str]]:
    """Zwraca ID rekordow glownych i elementow wraz z opisem polozenia."""

    if not isinstance(data, list):
        return
    for record_index, record in enumerate(data):
        if not isinstance(record, dict):
            continue
        record_id = get_record_id(record)
        if record_id:
            yield record_id, f"rekord[{record_index}]"
        elements = record.get("elementy")
        if not isinstance(elements, list):
            continue
        for element_index, element in enumerate(elements):
            if not isinstance(element, dict):
                continue
            element_id = get_record_id(element)
            if element_id:
                yield (
                    element_id,
                    f"rekord[{record_index}].elementy[{element_index}]",
                )


def duplicate_record_ids(data: Any) -> dict[str, list[str]]:
    occurrences: dict[str, list[str]] = {}
    for record_id, location in iter_record_ids(data):
        occurrences.setdefault(record_id, []).append(location)
    return {
        record_id: locations
        for record_id, locations in occurrences.items()
        if len(locations) > 1
    }


def letter_suffix_to_number(suffix: str) -> int:
    number = 0
    for character in suffix:
        number = number * 26 + (ord(character) - ord("a") + 1)
    return number


def number_to_letter_suffix(number: int) -> str:
    characters: list[str] = []
    while number > 0:
        number, remainder = divmod(number - 1, 26)
        characters.append(chr(ord("a") + remainder))
    return "".join(reversed(characters))


def lettered_id_namespaces(parent_id: str) -> Iterator[str]:
    """Generuje warianty `_a`, `_b`, ..., zachowujac istniejacy sufiks."""

    match = re.fullmatch(r"(.+)_([a-z]+)", parent_id)
    if match:
        base = match.group(1)
        number = letter_suffix_to_number(match.group(2)) + 1
    else:
        base = parent_id
        number = 1
    while True:
        yield f"{base}_{number_to_letter_suffix(number)}"
        number += 1


def assign_unique_element_ids(
    proposed_record: dict[str, Any],
    reserved_ids: set[str],
) -> dict[str, Any]:
    """Nadaje elementom jedna wolna przestrzen ID oparta na ID rodzica."""

    parent_id = get_record_id(proposed_record)
    elements = proposed_record.get("elementy")
    if not parent_id or not isinstance(elements, list):
        raise RuntimeError("Proponowany rekord zbiorczy nie ma ID lub listy elementow")
    if any(not isinstance(element, dict) for element in elements):
        raise RuntimeError(f"Niepoprawna lista elementow dla {parent_id}")

    def namespace_candidates() -> Iterator[str]:
        yield parent_id
        yield from lettered_id_namespaces(parent_id)

    selected_namespace = ""
    selected_ids: list[str] = []
    for namespace in namespace_candidates():
        candidate_ids = [
            f"{namespace}-{index:03d}"
            for index in range(1, len(elements) + 1)
        ]
        namespace_is_available = namespace == parent_id or namespace not in reserved_ids
        if namespace_is_available and not reserved_ids.intersection(candidate_ids):
            selected_namespace = namespace
            selected_ids = candidate_ids
            break

    assignments: list[dict[str, str]] = []
    for element, new_id in zip(elements, selected_ids):
        old_id = get_record_id(element)
        element["ID"] = new_id
        reserved_ids.add(new_id)
        assignments.append({"stare_ID": old_id, "nowe_ID": new_id})
    return {
        "parent_ID": parent_id,
        "przestrzen_ID_elementow": selected_namespace,
        "uzyto_wariantu_literowego": selected_namespace != parent_id,
        "elementy": assignments,
    }


def next_unused_lettered_id(identifier: str, unavailable: set[str]) -> str:
    for candidate in lettered_id_namespaces(identifier):
        if candidate not in unavailable:
            return candidate
    raise RuntimeError(f"Nie mozna przydzielic unikalnego ID dla {identifier}")


def make_record_ids_unique(data: list[Any]) -> list[dict[str, Any]]:
    """Zachowuje pierwsze ID, a kolejnym kolizjom nadaje wariant literowy."""

    original_ids = {record_id for record_id, _ in iter_record_ids(data)}
    used: set[str] = set()
    changes: list[dict[str, Any]] = []
    previous_parent_id = ""

    for record_index, record in enumerate(data):
        if not isinstance(record, dict):
            continue
        old_parent_id = get_record_id(record)
        if not old_parent_id:
            seed_id = previous_parent_id
            if not seed_id:
                for later_record in data[record_index + 1 :]:
                    if isinstance(later_record, dict) and get_record_id(later_record):
                        seed_id = get_record_id(later_record)
                        break
            if not seed_id:
                raise RuntimeError(
                    f"Nie mozna wyznaczyc ID dla rekordu[{record_index}]"
                )
            new_parent_id = next_unused_lettered_id(
                seed_id, original_ids | used
            )
            parent_was_renamed = True
            record["ID"] = new_parent_id
            changes.append(
                {
                    "rodzaj": "rekord_glowny",
                    "rekord_index": record_index,
                    "element_index": None,
                    "nazwa": record.get("nazwa"),
                    "stare_ID": "",
                    "nowe_ID": new_parent_id,
                    "powod": "brak_ID_w_zrodle",
                }
            )
        else:
            parent_was_renamed = old_parent_id in used

        if old_parent_id and parent_was_renamed:
            new_parent_id = next_unused_lettered_id(
                old_parent_id, original_ids | used
            )
            record["ID"] = new_parent_id
            changes.append(
                {
                    "rodzaj": "rekord_glowny",
                    "rekord_index": record_index,
                    "element_index": None,
                    "nazwa": record.get("nazwa"),
                    "stare_ID": old_parent_id,
                    "nowe_ID": new_parent_id,
                    "powod": "powtorzone_ID_w_zrodle",
                }
            )
        elif old_parent_id:
            new_parent_id = old_parent_id
        used.add(new_parent_id)
        previous_parent_id = new_parent_id

        elements = record.get("elementy")
        if not isinstance(elements, list):
            continue

        if parent_was_renamed:
            branch_namespace = new_parent_id
        else:
            branch_namespace = ""

        for element_index, element in enumerate(elements):
            if not isinstance(element, dict):
                continue
            old_element_id = get_record_id(element)
            if not parent_was_renamed and old_element_id and old_element_id not in used:
                used.add(old_element_id)
                continue

            position = element_index + 1
            if not branch_namespace:
                for namespace in lettered_id_namespaces(new_parent_id):
                    candidate = f"{namespace}-{position:03d}"
                    if namespace not in original_ids | used and candidate not in original_ids | used:
                        branch_namespace = namespace
                        break

            new_element_id = f"{branch_namespace}-{position:03d}"
            if new_element_id in original_ids | used:
                for namespace in lettered_id_namespaces(branch_namespace):
                    candidate = f"{namespace}-{position:03d}"
                    if namespace not in original_ids | used and candidate not in original_ids | used:
                        branch_namespace = namespace
                        new_element_id = candidate
                        break

            element["ID"] = new_element_id
            used.add(new_element_id)
            changes.append(
                {
                    "rodzaj": "element",
                    "rekord_index": record_index,
                    "element_index": element_index,
                    "nazwa": element.get("nazwa"),
                    "parent_ID": new_parent_id,
                    "stare_ID": old_element_id,
                    "nowe_ID": new_element_id,
                }
            )

    duplicates = duplicate_record_ids(data)
    if duplicates:
        raise RuntimeError(
            "Nie udalo sie usunac duplikatow ID: "
            + ", ".join(sorted(duplicates)[:20])
        )
    return changes


def apply_approved(
    input_path: Path,
    proposals_path: Path,
    approved_ids_path: Path,
    output_dir: Path,
) -> int:
    data = load_json(input_path)
    if not isinstance(data, list):
        raise RuntimeError("Tryb zastosowania wymaga listy rekordow w pliku JSON")
    source_duplicates = duplicate_record_ids(data)
    approved_ids = read_approved_ids(approved_ids_path)
    top_level_counts: dict[str, int] = {}
    for record in data:
        if isinstance(record, dict):
            record_id = get_record_id(record)
            if record_id:
                top_level_counts[record_id] = top_level_counts.get(record_id, 0) + 1
    ambiguous_source_ids = sorted(
        record_id
        for record_id in approved_ids
        if top_level_counts.get(record_id, 0) != 1
    )
    if ambiguous_source_ids:
        raise RuntimeError(
            "Zatwierdzone ID nie wskazuja jednoznacznie jednego rekordu "
            "glownego: " + ", ".join(ambiguous_source_ids)
        )
    proposal_groups: dict[str, list[dict[str, Any]]] = {}
    for row in load_jsonl(proposals_path):
        proposal_id = str(row.get("ID", "") or "").strip()
        if proposal_id:
            proposal_groups.setdefault(proposal_id, []).append(row)
    missing = sorted(approved_ids - set(proposal_groups))
    if missing:
        raise RuntimeError(f"Brak propozycji dla zatwierdzonych ID: {missing}")
    ambiguous = sorted(
        record_id
        for record_id in approved_ids
        if len(proposal_groups[record_id]) != 1
    )
    if ambiguous:
        raise RuntimeError(
            "Dla zatwierdzonych ID istnieje wiecej niz jedna propozycja: "
            + ", ".join(ambiguous)
        )
    proposals = {
        record_id: proposal_groups[record_id][0] for record_id in approved_ids
    }

    applied: list[str] = []
    id_assignments: list[dict[str, Any]] = []
    reserved_ids = {record_id for record_id, _ in iter_record_ids(data)}
    output_records: list[Any] = []
    for record in data:
        if not isinstance(record, dict) or get_record_id(record) not in approved_ids:
            output_records.append(record)
            continue
        record_id = get_record_id(record)
        proposal = proposals[record_id]
        if proposal.get("prompt_version") != PROMPT_VERSION:
            raise RuntimeError(
                f"Propozycja {record_id} pochodzi ze starszej wersji promptu; "
                "uruchom analize ponownie"
            )
        if proposal.get("decyzja") != "zbiorcze":
            raise RuntimeError(f"Zatwierdzono propozycje niezbiorcza: {record_id}")
        if proposal.get("record_sha256") != record_hash(record):
            raise RuntimeError(f"Rekord zrodlowy zmienil sie od analizy: {record_id}")
        proposed_record = proposal.get("proponowany_rekord")
        if not isinstance(proposed_record, dict):
            raise RuntimeError(f"Brak proponowanego rekordu: {record_id}")
        proposed_record = copy.deepcopy(proposed_record)
        id_assignments.append(
            assign_unique_element_ids(proposed_record, reserved_ids)
        )
        output_records.append(proposed_record)
        applied.append(record_id)

    not_found = sorted(approved_ids - set(applied))
    if not_found:
        raise RuntimeError(f"Nie znaleziono ID w pliku zrodlowym: {not_found}")

    normalized_id_changes = make_record_ids_unique(output_records)
    output_duplicates = duplicate_record_ids(output_records)
    if output_duplicates:
        examples = ", ".join(sorted(output_duplicates)[:20])
        suffix = " ..." if len(output_duplicates) > 20 else ""
        raise RuntimeError(
            "Po zastosowaniu propozycji nadal wystepuja powtarzajace sie ID; "
            f"plik nie zostal zapisany: {examples}{suffix}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{input_path.stem}_repaired.json"
    write_json(output_path, output_records)
    report = {
        "plik_zrodlowy": str(input_path),
        "plik_wynikowy": str(output_path),
        "zastosowane_ID": applied,
        "liczba_zastosowanych": len(applied),
        "przydzial_ID_elementow": id_assignments,
        "liczba_wariantow_literowych": sum(
            bool(item["uzyto_wariantu_literowego"])
            for item in id_assignments
        ),
        "duplikaty_ID_w_pliku_zrodlowym": source_duplicates,
        "liczba_zmienionych_istniejacych_ID": len(normalized_id_changes),
        "zmiany_istniejacych_ID": normalized_id_changes,
        "wszystkie_ID_unikalne": True,
        "czas_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output_dir / f"{input_path.stem}_apply_report.json", report)
    print(output_path)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Wykrywa indywidualne rekordy SGKP, ktore moga byc haslami "
            "zbiorczymi, i przygotowuje bezpieczne propozycje naprawy."
        )
    )
    parser.add_argument("input", type=Path, help="Zrodlowy plik JSON")
    parser.add_argument(
        "-o", "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--ollama-url", default=None)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--ids",
        nargs="+",
        default=None,
        help=(
            "Przetworz tylko wskazane ID kandydatow. ID mozna rozdzielac "
            "spacjami lub przecinkami."
        ),
    )
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES)
    parser.add_argument("--retry-delay", type=float, default=DEFAULT_RETRY_DELAY)
    parser.add_argument(
        "--settlement-types",
        type=Path,
        default=DEFAULT_SETTLEMENT_TYPES_PATH,
        help=(
            "Ekspercki CSV mapujacy pole typ na typ_punktu_osadniczego. "
            f"Domyslnie: {DEFAULT_SETTLEMENT_TYPES_PATH}"
        ),
    )
    parser.add_argument(
        "--abbreviations",
        type=Path,
        default=DEFAULT_ABBREVIATIONS_PATH,
        help=(
            "Lista skrotow SGKP dolaczana do promptu. "
            f"Domyslnie: {DEFAULT_ABBREVIATIONS_PATH}"
        ),
    )
    parser.add_argument(
        "--scan-only",
        action="store_true",
        help="Tylko zapisz liste kandydatow; nie wywoluj modelu.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rozpocznij tworzenie propozycji od nowa.",
    )
    parser.add_argument(
        "--apply-proposals",
        type=Path,
        default=None,
        help="Zastosuj zatwierdzone propozycje z podanego JSONL.",
    )
    parser.add_argument(
        "--approved-ids",
        type=Path,
        default=None,
        help="Plik tekstowy z jednym zatwierdzonym ID w wierszu.",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers musi byc dodatnie")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit musi byc dodatni")
    if args.retries < 1:
        parser.error("--retries musi byc dodatnie")
    if bool(args.apply_proposals) != bool(args.approved_ids):
        parser.error("--apply-proposals i --approved-ids musza byc podane razem")
    return args


def main() -> int:
    started_at = time.monotonic()
    args = parse_args()
    if not args.input.exists():
        print(f"Nie znaleziono pliku: {args.input}", file=sys.stderr)
        return 2
    if args.apply_proposals:
        return apply_approved(
            args.input,
            args.apply_proposals,
            args.approved_ids,
            args.output_dir,
        )

    data = load_json(args.input)
    candidates = find_candidates(data)
    all_candidate_count = len(candidates)
    if args.ids:
        requested_ids = {
            item.strip()
            for value in args.ids
            for item in value.split(",")
            if item.strip()
        }
        available_ids = {
            get_record_id(candidate["record"]) for candidate in candidates
        }
        missing_ids = sorted(requested_ids - available_ids)
        if missing_ids:
            print(
                "Wskazane ID nie sa kandydatami: " + ", ".join(missing_ids),
                file=sys.stderr,
            )
            return 2
        candidates = [
            candidate
            for candidate in candidates
            if get_record_id(candidate["record"]) in requested_ids
        ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_dir / f"{args.input.stem}_collective_repair"
    candidates_csv = prefix.with_suffix(".candidates.csv")
    proposals_jsonl = prefix.with_suffix(".proposals.jsonl")
    proposals_csv = prefix.with_suffix(".proposals.csv")
    errors_jsonl = prefix.with_suffix(".errors.jsonl")
    summary_json = prefix.with_suffix(".summary.json")
    write_candidate_csv(candidates_csv, candidates)
    print(
        f"Rekordy indywidualne={sum(1 for _ in iter_individual_records(data))}, "
        f"kandydaci={all_candidate_count}, wybrani={len(candidates)}",
        file=sys.stderr,
    )
    if args.scan_only:
        print(candidates_csv)
        return 0

    load_env_files(args.input)
    model = args.model or os.environ.get("OLLAMA_MODEL", DEFAULT_MODEL)
    base_url = args.ollama_url or os.environ.get("OLLAMA_URL", DEFAULT_OLLAMA_URL)
    api_key = os.environ.get("OLLAMA_API_KEY")
    if not args.settlement_types.exists():
        print(
            f"Nie znaleziono katalogu typow osadniczych: {args.settlement_types}",
            file=sys.stderr,
        )
        return 2
    if not args.abbreviations.exists():
        print(
            f"Nie znaleziono listy skrotow SGKP: {args.abbreviations}",
            file=sys.stderr,
        )
        return 2
    settlement_type_mapping = load_settlement_type_mapping(args.settlement_types)
    abbreviations_text, abbreviation_mapping = load_abbreviations(
        args.abbreviations
    )

    if args.overwrite:
        write_jsonl(proposals_jsonl, [])
        write_jsonl(errors_jsonl, [])
    previous_rows = load_jsonl(proposals_jsonl)
    existing = {str(row.get("validation_key", "") or ""): row for row in previous_rows}
    previous_error_rows = load_jsonl(errors_jsonl)
    existing_errors = {
        str(row.get("validation_key", "") or ""): row
        for row in previous_error_rows
        if row.get("validation_key")
    }
    pending: list[dict[str, Any]] = []
    for candidate in candidates:
        key = validation_key(args.input.name, candidate["record"], model)
        if key not in existing:
            pending.append(candidate)
    if args.limit is not None:
        pending = pending[: args.limit]

    completed = 0
    errors = 0
    if args.workers == 1:
        for index, candidate in enumerate(pending, start=1):
            record = candidate["record"]
            print(
                f"[{index}/{len(pending)}] {record_label(record)}",
                file=sys.stderr,
            )
            try:
                row = make_proposal_row(
                    args.input,
                    candidate,
                    base_url,
                    api_key,
                    model,
                    args.timeout,
                    args.retries,
                    args.retry_delay,
                    settlement_type_mapping,
                    abbreviation_mapping,
                    abbreviations_text,
                )
            except Exception as exc:
                errors += 1
                error_row = make_error_row(args.input, record, model, exc)
                existing_errors[error_row["validation_key"]] = error_row
                print(
                    f"[{index}/{len(pending)}] Blad {record_label(record)}: "
                    f"{short_error(exc)}",
                    file=sys.stderr,
                )
            else:
                append_jsonl(proposals_jsonl, row)
                existing[row["validation_key"]] = row
                existing_errors.pop(row["validation_key"], None)
                completed += 1
                print(
                    f"[{index}/{len(pending)}] Gotowe {record_label(record)}",
                    file=sys.stderr,
                )
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    make_proposal_row,
                    args.input,
                    candidate,
                    base_url,
                    api_key,
                    model,
                    args.timeout,
                    args.retries,
                    args.retry_delay,
                    settlement_type_mapping,
                    abbreviation_mapping,
                    abbreviations_text,
                ): candidate
                for candidate in pending
            }
            processed = 0
            for future in as_completed(futures):
                processed += 1
                candidate = futures[future]
                record = candidate["record"]
                try:
                    row = future.result()
                except Exception as exc:
                    errors += 1
                    error_row = make_error_row(args.input, record, model, exc)
                    existing_errors[error_row["validation_key"]] = error_row
                    print(
                        f"[{processed}/{len(pending)}] Blad "
                        f"{record_label(record)}: {short_error(exc)}",
                        file=sys.stderr,
                    )
                else:
                    append_jsonl(proposals_jsonl, row)
                    existing[row["validation_key"]] = row
                    existing_errors.pop(row["validation_key"], None)
                    completed += 1
                    print(
                        f"[{processed}/{len(pending)}] Gotowe "
                        f"{record_label(record)}",
                        file=sys.stderr,
                    )

    current_keys = {
        validation_key(args.input.name, candidate["record"], model)
        for candidate in candidates
    }
    final_rows = [existing[key] for key in current_keys if key in existing]
    final_rows.sort(key=lambda row: str(row.get("ID", "")))
    final_error_rows = [
        existing_errors[key] for key in current_keys if key in existing_errors
    ]
    final_error_rows.sort(key=lambda row: str(row.get("ID", "")))
    write_jsonl(proposals_jsonl, final_rows)
    write_jsonl(errors_jsonl, final_error_rows)
    write_proposal_csv(proposals_csv, final_rows)
    summary = {
        "plik_zrodlowy": str(args.input),
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "katalog_typow_osadniczych": str(args.settlement_types),
        "lista_skrotow": str(args.abbreviations),
        "kandydaci": len(candidates),
        "nowe_zapytania": len(pending),
        "nowe_wyniki": completed,
        "bledy": errors,
        "nowe_bledy": errors,
        "bledy_lacznie": len(final_error_rows),
        "wyniki_lacznie": len(final_rows),
        "decyzje": {
            decision: sum(row.get("decyzja") == decision for row in final_rows)
            for decision in ("zbiorcze", "indywidualne", "niepewne")
        },
        "wymagaja_szczegolnej_kontroli": sum(
            bool(row.get("wymaga_szczegolnej_kontroli")) for row in final_rows
        ),
        "czas_calkowity": format_duration(time.monotonic() - started_at),
        "pliki": {
            "kandydaci_csv": str(candidates_csv),
            "propozycje_jsonl": str(proposals_jsonl),
            "propozycje_csv": str(proposals_csv),
            "bledy_jsonl": str(errors_jsonl),
        },
    }
    write_json(summary_json, summary)
    print(proposals_jsonl)
    print(proposals_csv)
    print(summary_json)
    print(f"Czas wykonania: {summary['czas_calkowity']}", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
