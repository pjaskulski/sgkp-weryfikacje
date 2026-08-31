#!/usr/bin/env python3
"""Uzupelnia typy miejscowosci w haslach oznaczonych jako odsyłacze.

Skrypt analizuje rekordy indywidualne i elementy rekordow zbiorczych przez
Gemme udostepniona w Ollamie. Zachowuje typ ``odsyłacz``, a dla rekordow,
ktore mimo odeslania zawieraja jawny opis miejscowosci, dodaje typy z tekstu
oraz wylicza ``typ_punktu_osadniczego`` z katalogu eksperckiego.

Plik wejsciowy nigdy nie jest modyfikowany. Wynik, decyzje modelu, bledy i
podsumowanie sa zapisywane do osobnych plikow.
"""

from __future__ import annotations

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
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
import unicodedata
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_MODEL = "gemma4:31b-cloud"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_OUTPUT_DIR = Path("sgkp_naprawa/odsyłacze")
DEFAULT_WORKERS = 1
DEFAULT_TIMEOUT = 300.0
DEFAULT_RETRIES = 5
DEFAULT_RETRY_DELAY = 30.0
DEFAULT_LONG_TEXT_THRESHOLD = 120
BASE_PATH = Path(__file__).resolve().parents[2]
DEFAULT_SETTLEMENT_TYPES_PATH = (
    BASE_PATH
    / "sgkp_information_extraction"
    / "dictionary"
    / "typy_punktow_osadniczych_v2.csv"
)
DEFAULT_ABBREVIATIONS_PATH = (
    BASE_PATH
    / "sgkp_information_extraction"
    / "dictionary"
    / "prompt_sgkp_skroty.txt"
)
PROMPT_VERSION = "reference_type_repair_ollama_v1"
TRANSIENT_HTTP_CODES = {408, 425, 429, 500, 502, 503, 504}
DECISIONS = {
    "dodaj_typ_miejscowosci",
    "brak_podstaw",
    "nie_miejscowosc",
    "niepewne",
}
CONFIDENCE_LEVELS = {"wysoka", "srednia", "niska"}

ADMIN_TEXT_RE = re.compile(
    r"(?i)(?<!\w)(?:pow|gm|gub)(?=\s|[.,:])[.,]?"
)
STATISTICS_TEXT_RE = re.compile(
    r"(?i)(?<!\w)(?:dm|dym|mk|mieszk\w*|ludn\w*)(?=\s|[.,:])[.,]?"
)
SETTLEMENT_TEXT_RE = re.compile(
    r"(?ix)(?<!\w)(?:"
    r"wś|wieś\w*|folw\w*[.]?|os[.]|osad\w*|mko|mczko|mstko|msto|miast\w*|"
    r"kol[.]|koloni\w*|zaśc\w*[.]?|przys\w*[.]?|chut\w*[.]?|futor\w*|"
    r"słobod\w*|przedmieś\w*|leśnicz\w*|gajów\w*|karczm\w*|młyn\w*|"
    r"cegiel\w*|browar\w*|hut\w*|kopalni\w*|dobra|dominium|mająt\w*|"
    r"posiadłoś\w*|dwór|budy|buda|st[.]\s*(?:p[.]|dr[.]\s*ż[.])|"
    r"stac\w*\s+(?:poczt\w*|kolej\w*)|przystan\w*|przystanek\w*"
    r")(?!\w)"
)
STRUCTURED_SIGNAL_FIELDS = {
    "powiat_ocr",
    "gmina",
    "gubernia",
    "l_mk_statystyka",
    "l_dm_statystyka",
    "ludność_wyznanie",
}


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


def write_json(path: Path, data: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")
    temporary.replace(path)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False))
        handle.write("\n")


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    value = value.replace("„", '"').replace("”", '"').replace("’", "'")
    return " ".join(value.split())


def word_tokens(value: str) -> list[str]:
    return re.findall(r"[^\W\d_]+", normalized_text(value), flags=re.UNICODE)


def words_roughly_match(left: str, right: str) -> bool:
    if left == right:
        return True
    shortest = min(len(left), len(right))
    if shortest < 4:
        return False
    common = 0
    for left_char, right_char in zip(left, right):
        if left_char != right_char:
            break
        common += 1
    return common >= max(4, shortest - 3)


def has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().casefold() not in {"", "/", "null", "none", "brak"}
    if isinstance(value, (list, dict, tuple, set)):
        return bool(value)
    return True


def string_list(value: Any, field: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"Pole {field} nie jest lista napisow")
    return [item.strip() for item in value if item.strip()]


def unique_strings(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = " ".join(str(value).split()).strip()
        key = cleaned.casefold()
        if cleaned and key not in seen:
            seen.add(key)
            result.append(cleaned)
    return result


def contains_reference_type(value: Any) -> bool:
    try:
        types = string_list(value, "typ")
    except ValueError:
        return False
    return any(item.casefold() == "odsyłacz" for item in types)


def load_settlement_type_mapping(path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"typ_model", "typ_punktu_osadniczego"}
        if not reader.fieldnames or not required <= set(reader.fieldnames):
            raise RuntimeError(f"Niepoprawny katalog typow: {path}")
        for row in reader:
            source = str(row.get("typ_model", "") or "").strip()
            target = str(row.get("typ_punktu_osadniczego", "") or "").strip()
            if source and target:
                mapping[source.casefold()] = target
    return mapping


def load_abbreviations(path: Path) -> tuple[str, dict[str, str]]:
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


def normalize_entry_type(value: str, abbreviations: dict[str, str]) -> str:
    cleaned = " ".join(value.split()).strip()
    return " ".join(abbreviations.get(cleaned.casefold(), cleaned).split()).strip()


def settlement_types_for(
    entry_types: list[str],
    mapping: dict[str, str],
) -> tuple[list[str], list[str], list[str]]:
    settlement_types: list[str] = []
    unknown: list[str] = []
    not_applicable: list[str] = []
    for entry_type in entry_types:
        mapped = mapping.get(entry_type.casefold())
        if mapped is None:
            unknown.append(entry_type)
        elif mapped.casefold() == "nie dotyczy":
            not_applicable.append(entry_type)
        elif mapped not in settlement_types:
            settlement_types.append(mapped)
    return settlement_types, unknown, not_applicable


def task_label(task: dict[str, Any]) -> str:
    name = " ".join(str(task.get("nazwa", "") or "").split())
    return f"{task['ID']} {name}".strip()


def format_duration(seconds: float) -> str:
    total = int(round(seconds))
    minutes, secs = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def short_error(error: Exception) -> str:
    message = " ".join((str(error).strip() or error.__class__.__name__).split())
    return message if len(message) <= 500 else message[:497] + "..."


def text_scoped_to_element(elements: list[Any], element_index: int) -> str:
    """Usuwa doklejona kopie kolejnego elementu tylko z kontekstu modelu."""

    current = elements[element_index]
    if not isinstance(current, dict):
        return ""
    text = str(current.get("text", "") or "").strip()
    cuts: list[int] = []
    for later in elements[element_index + 1 :]:
        if not isinstance(later, dict):
            continue
        later_text = str(later.get("text", "") or "").strip()
        if len(later_text) < 10:
            continue
        position = text.find(later_text)
        if position > 0:
            cuts.append(position)
    return text[: min(cuts)].rstrip() if cuts else text


def candidate_reasons(target: dict[str, Any], text: str, threshold: int) -> list[str]:
    reasons: list[str] = []
    if SETTLEMENT_TEXT_RE.search(text):
        reasons.append("sygnal_typu_w_tekscie")
    if ADMIN_TEXT_RE.search(text):
        reasons.append("dane_administracyjne_w_tekscie")
    if STATISTICS_TEXT_RE.search(text):
        reasons.append("statystyka_w_tekscie")
    structured = sorted(
        field for field in STRUCTURED_SIGNAL_FIELDS if has_value(target.get(field))
    )
    if structured:
        reasons.append("istniejace_pola:" + ",".join(structured))
    if len(text) >= threshold:
        reasons.append("dluzszy_opis")
    return reasons


def collect_tasks(
    data: Any,
    all_references: bool,
    long_text_threshold: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    if not isinstance(data, list):
        raise RuntimeError("Plik wejsciowy powinien zawierac liste rekordow")
    all_reference_tasks: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    stats = {
        "cele_lacznie": 0,
        "odsyłacze_lacznie": 0,
        "pominiete_bez_tekstu": 0,
        "pominiete_bez_sygnalu": 0,
        "kandydaci": 0,
    }

    def add(
        target: dict[str, Any],
        record_index: int,
        element_index: int | None,
        parent: dict[str, Any] | None,
        scoped_text: str | None = None,
    ) -> None:
        stats["cele_lacznie"] += 1
        if not contains_reference_type(target.get("typ")):
            return
        stats["odsyłacze_lacznie"] += 1
        text = scoped_text if scoped_text is not None else str(target.get("text", "") or "").strip()
        if not text:
            stats["pominiete_bez_tekstu"] += 1
            return
        record_id = str(target.get("ID", "") or "").strip()
        if not record_id:
            raise RuntimeError(
                f"Brak ID dla rekordu {record_index}, elementu {element_index}"
            )
        if record_id in seen_ids:
            raise RuntimeError(f"Powtorzone ID odsyłacza: {record_id}")
        seen_ids.add(record_id)
        reasons = candidate_reasons(target, text, long_text_threshold)
        task = {
            "ID": record_id,
            "nazwa": target.get("nazwa"),
            "rodzaj_celu": "element" if element_index is not None else "indywidualne",
            "parent_ID": str(parent.get("ID", "") or "").strip() if parent else None,
            "record_index": record_index,
            "element_index": element_index,
            "text": text,
            "typ_przed": string_list(target.get("typ"), "typ"),
            "typ_punktu_przed": string_list(
                target.get("typ_punktu_osadniczego"),
                "typ_punktu_osadniczego",
            ),
            "powody_kandydatury": reasons,
        }
        all_reference_tasks.append(task)
        if all_references or reasons:
            candidates.append(task)
            stats["kandydaci"] += 1
        else:
            stats["pominiete_bez_sygnalu"] += 1

    for record_index, record in enumerate(data):
        if not isinstance(record, dict):
            continue
        if record.get("rodzaj") == "indywidualne":
            add(record, record_index, None, None)
        elif record.get("rodzaj") == "zbiorcze":
            elements = record.get("elementy", [])
            if not isinstance(elements, list):
                continue
            for element_index, element in enumerate(elements):
                if isinstance(element, dict):
                    add(
                        element,
                        record_index,
                        element_index,
                        record,
                        text_scoped_to_element(elements, element_index),
                    )
    return candidates, all_reference_tasks, stats


def task_fingerprint(task: dict[str, Any]) -> str:
    return sha256_text(
        canonical_json(
            {
                "ID": task["ID"],
                "nazwa": task.get("nazwa"),
                "text": task["text"],
                "typ_przed": task["typ_przed"],
                "typ_punktu_przed": task["typ_punktu_przed"],
            }
        )
    )


def validation_key(
    source_name: str,
    task: dict[str, Any],
    model: str,
    resources_hash: str,
) -> str:
    return sha256_text(
        canonical_json(
            {
                "source": source_name,
                "task_fingerprint": task_fingerprint(task),
                "model": model,
                "prompt_version": PROMPT_VERSION,
                "resources_hash": resources_hash,
            }
        )
    )


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
                    "Jestes asystentem historyka analizujacym tekst SGKP. "
                    "Nie zgadujesz i odpowiadasz wylacznie poprawnym JSON-em."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "format": "json",
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
        raise OllamaTransientError(
            f"Nie mozna polaczyc sie z Ollama: {base_url}"
        ) from exc

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
        if candidate.startswith("{{") and candidate.endswith("}}"):
            candidates.append(candidate[1:-1])
        if candidate.startswith("{{"):
            candidates.append(candidate[1:])
        if candidate.endswith("}}"):
            candidates.append(candidate[:-1])
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
    for candidate in candidates:
        try:
            parsed = ast.literal_eval(candidate)
        except (SyntaxError, ValueError, TypeError) as exc:
            errors.append(str(exc))
            continue
        if isinstance(parsed, dict):
            return parsed
        errors.append("element glowny zapisu tolerancyjnego nie jest obiektem")
    detail = errors[-1] if errors else "brak obiektu"
    raise ModelResponseError(f"Nie mozna odczytac JSON modelu: {detail}", output)


def build_prompt(task: dict[str, Any], abbreviations_text: str) -> str:
    return f"""Ocen rekord SGKP, ktoremu obecnie przypisano typ `odsyłacz`.

ID: {task['ID']}
Nazwa hasla: {task.get('nazwa')}
Rodzaj rekordu: {task['rodzaj_celu']}
Obecne typy: {json.dumps(task['typ_przed'], ensure_ascii=False)}

TEKST:
{task['text']}

CEL:
Ustal, czy haslo poza odeslaniem do innego artykulu jawnie opisuje glowny
obiekt jako miejscowosc lub punkt osadniczy i podaje jego typ, np. wieś,
folwark, osada, miasto, miasteczko, kolonia, zaścianek, przysiółek, chutor,
leśniczówka, młyn albo stacja pocztowa.

REGULY:
1. Typ `odsyłacz` pozostaje w danych. Wskazujesz tylko dodatkowe typy
   miejscowosci, ktore nalezy do niego dopisac.
2. Skrót `ob.` lub `zob.` sam w sobie oznacza odeslanie i nie wyklucza
   dodatkowego typu. Przykład `Aleksandrowo, folw., pow. pleszewski, ob.
   Klenka` jest jednoczesnie odsyłaczem i opisem folwarku.
3. `Anisiniowicze, wś, pow. nowogródzki, 200 mieszk., ob. Dryświaty` wymaga
   typu `wieś`. Dane administracyjne, liczba domow lub mieszkańców wzmacniaja
   pewnosc, ze opis dotyczy rzeczywistej miejscowosci.
4. Sam zapis `Nazwa, ob. InnaNazwa`, lista wariantow nazw albo tlumaczenie
   nazwy bez typu i bez realnego opisu obiektu nie wystarcza. Wybierz wtedy
   `brak_podstaw`.
5. Haslo moze byc rozbudowanym odsyłaczem do rzeki, jeziora, gory, pasma,
   krainy albo innego obiektu nieosadniczego. Wtedy wybierz
   `nie_miejscowosc` i nie zwracaj typu miejscowosci.
6. Typ i dane musza dotyczyc glownego obiektu o nazwie podanej w `Nazwa
   hasla`. Nie przenos typu obiektu wymienionego dopiero po `ob.` ani typu
   sasiedniej lub nadrzednej miejscowosci.
7. Nie wyprowadzaj typu jedynie z powiatu, gminy, parafii lub liczby ludnosci.
   Ogolny typ `miejscowość` wolno zwrocic tylko wtedy, gdy to slowo wystepuje
   jawnie w tekscie. W przeciwnym razie wybierz `brak_podstaw` albo
   `niepewne`.
8. Typy zwracaj jako pelne nazwy, nigdy jako skroty: `wś` -> `wieś`, `folw.`
   -> `folwark`, `mko` -> `miasteczko`, `os.` -> `osada`.
9. Gdy tekst podaje kilka typow glownego obiektu, zwroc je osobno, np.
   `wś i folw.` -> `wieś` oraz `folwark`. Nie lacz ich w jeden napis.
10. Nie zwracaj `odsyłacz` w `typy_miejscowosci`; ten typ juz istnieje.
11. Kazdy `dowod` ma byc krotkim, doslownym cytatem z TEKSTU zawierajacym
    oznaczenie danego typu. Nie parafrazuj i nie cytuj samej nazwy hasla.
12. Wybierz wysoka pewnosc tylko dla typu podanego wprost. Srednia jest
    dopuszczalna dla ogolnego `miejscowość`. Przy sprzecznosci wybierz
    `niepewne`.

SLOWNIK SKROTOW SGKP:
{abbreviations_text}

Dozwolone decyzje: `dodaj_typ_miejscowosci`, `brak_podstaw`,
`nie_miejscowosc`, `niepewne`.
Dozwolona pewnosc: `wysoka`, `srednia`, `niska`.

Zwroc wylacznie obiekt JSON:
{{
  "decyzja": "dodaj_typ_miejscowosci",
  "pewnosc": "wysoka",
  "typy_miejscowosci": [
    {{"typ": "wieś", "dowod": "wś"}},
    {{"typ": "folwark", "dowod": "folw."}}
  ],
  "uzasadnienie": "Tekst jawnie okresla typ glownego obiektu."
}}

Dla decyzji innych niz `dodaj_typ_miejscowosci` zwroc pusta liste
`typy_miejscowosci`."""


def type_supported_by_evidence(
    entry_type: str,
    evidence: str,
    abbreviations: dict[str, str],
) -> bool:
    normalized_type = normalized_text(entry_type)
    normalized_evidence = normalized_text(evidence)
    if normalized_type in normalized_evidence:
        return True
    type_words = word_tokens(entry_type)
    evidence_words = word_tokens(evidence)
    if any(
        words_roughly_match(type_word, evidence_word)
        for type_word in type_words
        for evidence_word in evidence_words
    ):
        return True
    for abbreviation, expansion in abbreviations.items():
        if normalized_text(expansion) != normalized_type:
            continue
        if normalized_text(abbreviation) in normalized_evidence:
            return True
    return False


def normalize_response(
    parsed: dict[str, Any],
    task: dict[str, Any],
    settlement_mapping: dict[str, str],
    abbreviations: dict[str, str],
) -> dict[str, Any]:
    if isinstance(parsed.get("wynik"), dict):
        parsed = parsed["wynik"]
    decision = str(parsed.get("decyzja", "") or "").strip().casefold()
    aliases = {
        "dodaj_typ": "dodaj_typ_miejscowosci",
        "odsyłacz_i_miejscowość": "dodaj_typ_miejscowosci",
        "odsyłacz_i_miejscowosc": "dodaj_typ_miejscowosci",
        "czysty_odsyłacz": "brak_podstaw",
        "brak": "brak_podstaw",
        "nie_dotyczy": "nie_miejscowosc",
        "nie_miejscowość": "nie_miejscowosc",
    }
    decision = aliases.get(decision, decision)
    if decision not in DECISIONS:
        raise ValueError(f"Niepoprawna decyzja: {decision!r}")
    confidence = str(parsed.get("pewnosc", "") or "").strip().casefold()
    confidence_aliases = {
        "wysoki": "wysoka",
        "średnia": "srednia",
        "średni": "srednia",
        "sredni": "srednia",
        "niski": "niska",
    }
    confidence = confidence_aliases.get(confidence, confidence)
    if confidence not in CONFIDENCE_LEVELS:
        raise ValueError(f"Niepoprawna pewnosc: {confidence!r}")
    reasoning = str(parsed.get("uzasadnienie", "") or "").strip()
    raw_types = parsed.get("typy_miejscowosci", [])
    if raw_types is None:
        raw_types = []
    if not isinstance(raw_types, list):
        raise ValueError("typy_miejscowosci nie jest lista")

    if decision != "dodaj_typ_miejscowosci":
        if raw_types:
            raise ValueError(
                f"Decyzja {decision} wymaga pustej listy typy_miejscowosci"
            )
        return {
            "decyzja": decision,
            "pewnosc": confidence,
            "typy_miejscowosci": [],
            "typy_punktu_osadniczego": [],
            "uzasadnienie": reasoning,
        }

    if not raw_types:
        raise ValueError("Decyzja dodaj_typ_miejscowosci wymaga co najmniej jednego typu")
    normalized_types: list[dict[str, str]] = []
    seen_types: set[str] = set()
    for item in raw_types:
        if not isinstance(item, dict):
            raise ValueError("Element typy_miejscowosci nie jest obiektem")
        raw_type = str(item.get("typ", "") or "").strip()
        evidence = str(item.get("dowod", "") or "").strip()
        if not raw_type or not evidence:
            raise ValueError("Brak typu lub dowodu w typy_miejscowosci")
        entry_type = normalize_entry_type(raw_type, abbreviations)
        if entry_type.casefold() == "odsyłacz":
            raise ValueError("Nie wolno zwracac typu odsyłacz jako typu miejscowosci")
        if normalized_text(evidence) not in normalized_text(task["text"]):
            raise ValueError(f"Dowod dla typu {entry_type} nie wystepuje w tekscie")
        if not type_supported_by_evidence(entry_type, evidence, abbreviations):
            raise ValueError(f"Dowod nie potwierdza typu {entry_type}")
        settlement_types, unknown, not_applicable = settlement_types_for(
            [entry_type], settlement_mapping
        )
        if unknown:
            raise ValueError(f"Typ nie wystepuje w katalogu eksperckim: {entry_type}")
        if not_applicable or not settlement_types:
            raise ValueError(f"Typ nie jest typem punktu osadniczego: {entry_type}")
        key = entry_type.casefold()
        if key not in seen_types:
            seen_types.add(key)
            normalized_types.append({"typ": entry_type, "dowod": evidence})

    settlement_types, _, _ = settlement_types_for(
        [item["typ"] for item in normalized_types], settlement_mapping
    )
    return {
        "decyzja": decision,
        "pewnosc": confidence,
        "typy_miejscowosci": normalized_types,
        "typy_punktu_osadniczego": settlement_types,
        "uzasadnienie": reasoning,
    }


def extract_with_retries(
    base_url: str,
    api_key: str | None,
    model: str,
    task: dict[str, Any],
    abbreviations_text: str,
    settlement_mapping: dict[str, str],
    abbreviations: dict[str, str],
    timeout: float,
    retries: int,
    retry_delay: float,
) -> dict[str, Any]:
    prompt = build_prompt(task, abbreviations_text)
    current_prompt = prompt
    for attempt in range(1, max(1, retries) + 1):
        output = ""
        try:
            output = ollama_chat(base_url, api_key, model, current_prompt, timeout)
            return normalize_response(
                parse_model_json(output), task, settlement_mapping, abbreviations
            )
        except OllamaTransientError as exc:
            if attempt == retries:
                raise
            print(
                f"{task_label(task)}: blad tymczasowy ({attempt}/{retries}): "
                f"{short_error(exc)}",
                file=sys.stderr,
            )
            time.sleep(max(0.0, retry_delay))
        except ValueError as exc:
            if attempt == retries:
                raw_output = getattr(exc, "raw_output", output)
                raise ModelResponseError(short_error(exc), raw_output) from exc
            print(
                f"{task_label(task)}: bledny wynik modelu ({attempt}/{retries}): "
                f"{short_error(exc)}",
                file=sys.stderr,
            )
            current_prompt = (
                prompt
                + "\n\nPOPRZEDNIA ODPOWIEDZ BYLA NIEPOPRAWNA. "
                + f"Walidator zglosil: {short_error(exc)}. "
                + "Zwroc caly poprawiony obiekt JSON bez komentarza.\n"
                + output
            )
            time.sleep(min(5.0, max(0.0, retry_delay)))
    raise RuntimeError("Nieudana odpowiedz modelu")


def make_result_row(
    source_path: Path,
    task: dict[str, Any],
    base_url: str,
    api_key: str | None,
    model: str,
    resources_hash: str,
    abbreviations_text: str,
    settlement_mapping: dict[str, str],
    abbreviations: dict[str, str],
    timeout: float,
    retries: int,
    retry_delay: float,
) -> dict[str, Any]:
    result = extract_with_retries(
        base_url,
        api_key,
        model,
        task,
        abbreviations_text,
        settlement_mapping,
        abbreviations,
        timeout,
        retries,
        retry_delay,
    )
    return {
        "plik_zrodlowy": source_path.name,
        "ID": task["ID"],
        "nazwa": task.get("nazwa"),
        "rodzaj_celu": task["rodzaj_celu"],
        "parent_ID": task.get("parent_ID"),
        "record_index": task["record_index"],
        "element_index": task.get("element_index"),
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "validation_key": validation_key(
            source_path.name, task, model, resources_hash
        ),
        "task_fingerprint": task_fingerprint(task),
        "powody_kandydatury": task["powody_kandydatury"],
        "typ_przed": task["typ_przed"],
        "typ_punktu_przed": task["typ_punktu_przed"],
        **result,
        "czas_utc": datetime.now(timezone.utc).isoformat(),
    }


def make_error_row(
    source_path: Path,
    task: dict[str, Any],
    model: str,
    resources_hash: str,
    error: Exception,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "plik_zrodlowy": source_path.name,
        "ID": task["ID"],
        "nazwa": task.get("nazwa"),
        "rodzaj_celu": task["rodzaj_celu"],
        "parent_ID": task.get("parent_ID"),
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "validation_key": validation_key(
            source_path.name, task, model, resources_hash
        ),
        "task_fingerprint": task_fingerprint(task),
        "blad": short_error(error),
        "czas_utc": datetime.now(timezone.utc).isoformat(),
    }
    raw_output = getattr(error, "raw_output", "")
    if raw_output:
        row["surowa_odpowiedz_modelu"] = raw_output
    return row


def write_candidates_csv(path: Path, tasks: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    columns = [
        "ID",
        "nazwa",
        "rodzaj_celu",
        "parent_ID",
        "typ_przed",
        "typ_punktu_przed",
        "powody_kandydatury",
        "dlugosc_tekstu",
    ]
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for task in tasks:
            writer.writerow(
                {
                    "ID": task["ID"],
                    "nazwa": task.get("nazwa"),
                    "rodzaj_celu": task["rodzaj_celu"],
                    "parent_ID": task.get("parent_ID"),
                    "typ_przed": " | ".join(task["typ_przed"]),
                    "typ_punktu_przed": " | ".join(task["typ_punktu_przed"]),
                    "powody_kandydatury": " | ".join(task["powody_kandydatury"]),
                    "dlugosc_tekstu": len(task["text"]),
                }
            )
    temporary.replace(path)


def write_results_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    columns = [
        "ID",
        "nazwa",
        "rodzaj_celu",
        "parent_ID",
        "decyzja",
        "pewnosc",
        "typ_przed",
        "typy_rozpoznane",
        "typ_po",
        "typ_punktu_przed",
        "typ_punktu_rozpoznany",
        "typ_punktu_po",
        "dowody",
        "uzasadnienie",
        "czy_zmieniono",
    ]
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            type_results = row.get("typy_miejscowosci", [])
            writer.writerow(
                {
                    "ID": row.get("ID"),
                    "nazwa": row.get("nazwa"),
                    "rodzaj_celu": row.get("rodzaj_celu"),
                    "parent_ID": row.get("parent_ID"),
                    "decyzja": row.get("decyzja"),
                    "pewnosc": row.get("pewnosc"),
                    "typ_przed": " | ".join(row.get("typ_przed", [])),
                    "typy_rozpoznane": " | ".join(
                        item.get("typ", "") for item in type_results
                    ),
                    "typ_po": " | ".join(row.get("typ_po", [])),
                    "typ_punktu_przed": " | ".join(
                        row.get("typ_punktu_przed", [])
                    ),
                    "typ_punktu_rozpoznany": " | ".join(
                        row.get("typy_punktu_osadniczego", [])
                    ),
                    "typ_punktu_po": " | ".join(
                        row.get("typ_punktu_po", [])
                    ),
                    "dowody": " | ".join(
                        item.get("dowod", "") for item in type_results
                    ),
                    "uzasadnienie": row.get("uzasadnienie"),
                    "czy_zmieniono": row.get("czy_zmieniono", False),
                }
            )
    temporary.replace(path)


def resolve_target(data: list[Any], row: dict[str, Any]) -> dict[str, Any]:
    record = data[int(row["record_index"])]
    if not isinstance(record, dict):
        raise RuntimeError(f"Rekord dla {row['ID']} nie jest obiektem")
    element_index = row.get("element_index")
    if element_index is None:
        target = record
    else:
        elements = record.get("elementy")
        if not isinstance(elements, list):
            raise RuntimeError(f"Brak listy elementow dla {row['ID']}")
        target = elements[int(element_index)]
    if not isinstance(target, dict) or str(target.get("ID", "") or "") != row["ID"]:
        raise RuntimeError(f"Struktura pliku zmienila sie dla {row['ID']}")
    return target


def apply_results(
    data: Any,
    rows: list[dict[str, Any]],
    settlement_mapping: dict[str, str],
    apply_medium_confidence: bool,
) -> dict[str, int]:
    if not isinstance(data, list):
        raise RuntimeError("Plik wejsciowy powinien zawierac liste rekordow")
    counts = {"rekordy": 0, "typ": 0, "typ_punktu_osadniczego": 0}
    for row in rows:
        row["czy_zmieniono"] = False
        row["typ_po"] = list(row.get("typ_przed", []))
        row["typ_punktu_po"] = list(row.get("typ_punktu_przed", []))
        if row.get("decyzja") != "dodaj_typ_miejscowosci":
            continue
        confidence = row.get("pewnosc")
        if confidence != "wysoka" and not (
            confidence == "srednia" and apply_medium_confidence
        ):
            continue
        target = resolve_target(data, row)
        current_types = string_list(target.get("typ"), "typ")
        if current_types != row.get("typ_przed", []):
            continue
        recognized = [
            item.get("typ", "") for item in row.get("typy_miejscowosci", [])
        ]
        merged_types = unique_strings(current_types + recognized)
        current_points = string_list(
            target.get("typ_punktu_osadniczego"),
            "typ_punktu_osadniczego",
        )
        derived_points, _, _ = settlement_types_for(
            merged_types, settlement_mapping
        )
        merged_points = unique_strings(current_points + derived_points)
        type_changed = merged_types != current_types
        point_changed = merged_points != current_points
        if type_changed:
            target["typ"] = merged_types
            counts["typ"] += 1
        if point_changed:
            target["typ_punktu_osadniczego"] = merged_points
            counts["typ_punktu_osadniczego"] += 1
        if type_changed or point_changed:
            counts["rekordy"] += 1
            row["czy_zmieniono"] = True
        row["typ_po"] = merged_types
        row["typ_punktu_po"] = merged_points
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analizuje hasla typu odsyłacz przez Gemme/Ollama i dodaje jawnie "
            "potwierdzone typy miejscowosci."
        )
    )
    parser.add_argument("input", type=Path, help="Jeden zrodlowy plik sgkp_XX.json")
    parser.add_argument("-o", "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default=None)
    parser.add_argument("--ollama-url", default=None)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--ids",
        nargs="+",
        default=None,
        help="Przetworz tylko wskazane ID, rozdzielone spacjami lub przecinkami.",
    )
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES)
    parser.add_argument("--retry-delay", type=float, default=DEFAULT_RETRY_DELAY)
    parser.add_argument(
        "--settlement-types",
        type=Path,
        default=DEFAULT_SETTLEMENT_TYPES_PATH,
        help=f"Ekspercki katalog typow. Domyslnie: {DEFAULT_SETTLEMENT_TYPES_PATH}",
    )
    parser.add_argument(
        "--abbreviations",
        type=Path,
        default=DEFAULT_ABBREVIATIONS_PATH,
        help=f"Slownik skrotow SGKP. Domyslnie: {DEFAULT_ABBREVIATIONS_PATH}",
    )
    parser.add_argument(
        "--all-references",
        action="store_true",
        help="Analizuj wszystkie odsyłacze, takze bez wstepnego sygnalu opisu.",
    )
    parser.add_argument(
        "--long-text-threshold",
        type=int,
        default=DEFAULT_LONG_TEXT_THRESHOLD,
        help=(
            "Minimalna dlugosc tekstu uznawana za sygnal szerszego opisu. "
            f"Domyslnie: {DEFAULT_LONG_TEXT_THRESHOLD}."
        ),
    )
    parser.add_argument(
        "--apply-medium-confidence",
        action="store_true",
        help="Stosuj takze decyzje modelu o sredniej pewnosci.",
    )
    parser.add_argument(
        "--skip-failed",
        action="store_true",
        help="Nie ponawiaj zadan obecnych w aktualnym pliku errors.jsonl.",
    )
    parser.add_argument(
        "--scan-only",
        action="store_true",
        help="Tylko zapisz CSV kandydatow, bez wywolywania modelu.",
    )
    parser.add_argument(
        "--review-only",
        action="store_true",
        help="Uruchom model i raporty, ale nie stosuj zmian w kopii JSON.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rozpocznij wyniki i bledy od nowa.",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers musi byc dodatnie")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit musi byc dodatni")
    if args.retries < 1:
        parser.error("--retries musi byc dodatnie")
    if args.timeout <= 0:
        parser.error("--timeout musi byc dodatni")
    if args.long_text_threshold < 1:
        parser.error("--long-text-threshold musi byc dodatni")
    return args


def main() -> int:
    started_at = time.monotonic()
    args = parse_args()
    for path, label in (
        (args.input, "pliku JSON"),
        (args.settlement_types, "katalogu typow"),
        (args.abbreviations, "slownika skrotow"),
    ):
        if not path.is_file():
            print(f"Nie znaleziono {label}: {path}", file=sys.stderr)
            return 2
    try:
        data = load_json(args.input)
        candidates, all_references, selection_stats = collect_tasks(
            data,
            all_references=args.all_references,
            long_text_threshold=args.long_text_threshold,
        )
        settlement_mapping = load_settlement_type_mapping(args.settlement_types)
        abbreviations_text, abbreviations = load_abbreviations(args.abbreviations)
    except (OSError, json.JSONDecodeError, RuntimeError, ValueError) as exc:
        print(f"Nie mozna przygotowac danych: {short_error(exc)}", file=sys.stderr)
        return 2

    selected_tasks = candidates
    if args.ids:
        requested_ids = {
            item.strip()
            for value in args.ids
            for item in value.split(",")
            if item.strip()
        }
        reference_by_id = {task["ID"]: task for task in all_references}
        missing = sorted(requested_ids - set(reference_by_id))
        if missing:
            print(
                "Wskazane ID nie sa odsyłaczami: " + ", ".join(missing),
                file=sys.stderr,
            )
            return 2
        selected_tasks = [reference_by_id[item] for item in requested_ids]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_dir / f"{args.input.stem}_reference_type_repair"
    candidates_csv = prefix.with_suffix(".candidates.csv")
    results_jsonl = prefix.with_suffix(".results.jsonl")
    results_csv = prefix.with_suffix(".results.csv")
    errors_jsonl = prefix.with_suffix(".errors.jsonl")
    summary_json = prefix.with_suffix(".summary.json")
    output_json = args.output_dir / f"{args.input.stem}_reference_types_corrected.json"
    write_candidates_csv(candidates_csv, selected_tasks)
    print(
        f"Odsyłacze={selection_stats['odsyłacze_lacznie']}, "
        f"kandydaci={len(candidates)}, wybrani={len(selected_tasks)}",
        file=sys.stderr,
    )
    if args.scan_only:
        print(candidates_csv)
        return 0

    load_env_files(args.input)
    model = args.model or os.environ.get("OLLAMA_MODEL", DEFAULT_MODEL)
    base_url = args.ollama_url or os.environ.get("OLLAMA_URL", DEFAULT_OLLAMA_URL)
    api_key = os.environ.get("OLLAMA_API_KEY")
    resources_hash = sha256_text(
        file_sha256(args.settlement_types) + ":" + file_sha256(args.abbreviations)
    )

    if args.overwrite:
        write_jsonl(results_jsonl, [])
        write_jsonl(errors_jsonl, [])
    previous_rows = load_jsonl(results_jsonl)
    existing = {
        str(row.get("validation_key", "") or ""): row
        for row in previous_rows
        if row.get("validation_key")
    }
    previous_errors = load_jsonl(errors_jsonl)
    existing_errors = {
        str(row.get("validation_key", "") or ""): row
        for row in previous_errors
        if row.get("validation_key")
    }
    for key in existing:
        existing_errors.pop(key, None)

    pending: list[dict[str, Any]] = []
    for task in selected_tasks:
        key = validation_key(args.input.name, task, model, resources_hash)
        if key in existing:
            continue
        if args.skip_failed and key in existing_errors:
            continue
        pending.append(task)
    if args.limit is not None:
        pending = pending[: args.limit]
    print(
        f"Zapisane wyniki={len(existing)}, do przetworzenia={len(pending)}",
        file=sys.stderr,
    )

    completed = 0
    new_errors = 0

    def handle_success(row: dict[str, Any]) -> None:
        nonlocal completed
        append_jsonl(results_jsonl, row)
        existing[row["validation_key"]] = row
        existing_errors.pop(row["validation_key"], None)
        completed += 1

    def process(task: dict[str, Any]) -> dict[str, Any]:
        return make_result_row(
            args.input,
            task,
            base_url,
            api_key,
            model,
            resources_hash,
            abbreviations_text,
            settlement_mapping,
            abbreviations,
            args.timeout,
            args.retries,
            args.retry_delay,
        )

    if args.workers == 1:
        for processed, task in enumerate(pending, start=1):
            try:
                row = process(task)
            except Exception as exc:
                new_errors += 1
                error_row = make_error_row(
                    args.input, task, model, resources_hash, exc
                )
                existing_errors[error_row["validation_key"]] = error_row
                print(
                    f"[{processed}/{len(pending)}] Blad {task_label(task)}: "
                    f"{short_error(exc)}",
                    file=sys.stderr,
                )
            else:
                handle_success(row)
                recognized = ", ".join(
                    item["typ"] for item in row["typy_miejscowosci"]
                ) or "-"
                print(
                    f"[{processed}/{len(pending)}] Gotowe {task_label(task)}: "
                    f"{row['decyzja']} ({row['pewnosc']}) -> {recognized}",
                    file=sys.stderr,
                )
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process, task): task for task in pending}
            for processed, future in enumerate(as_completed(futures), start=1):
                task = futures[future]
                try:
                    row = future.result()
                except Exception as exc:
                    new_errors += 1
                    error_row = make_error_row(
                        args.input, task, model, resources_hash, exc
                    )
                    existing_errors[error_row["validation_key"]] = error_row
                    print(
                        f"[{processed}/{len(pending)}] Blad {task_label(task)}: "
                        f"{short_error(exc)}",
                        file=sys.stderr,
                    )
                else:
                    handle_success(row)
                    recognized = ", ".join(
                        item["typ"] for item in row["typy_miejscowosci"]
                    ) or "-"
                    print(
                        f"[{processed}/{len(pending)}] Gotowe {task_label(task)}: "
                        f"{row['decyzja']} ({row['pewnosc']}) -> {recognized}",
                        file=sys.stderr,
                    )

    current_tasks_by_id = {task["ID"]: task for task in candidates}
    for task in selected_tasks:
        current_tasks_by_id[task["ID"]] = task
    current_keys = {
        validation_key(args.input.name, task, model, resources_hash)
        for task in current_tasks_by_id.values()
    }
    final_rows = [existing[key] for key in current_keys if key in existing]
    final_rows.sort(key=lambda row: str(row.get("ID", "")))
    final_error_rows = [
        existing_errors[key]
        for key in current_keys
        if key in existing_errors and key not in existing
    ]
    final_error_rows.sort(key=lambda row: str(row.get("ID", "")))

    processed_data = copy.deepcopy(data)
    proposed_changes = apply_results(
        processed_data,
        final_rows,
        settlement_mapping,
        args.apply_medium_confidence,
    )
    if args.review_only:
        output_data = data
        applied_changes = {key: 0 for key in proposed_changes}
        for row in final_rows:
            row["czy_zmieniono"] = False
    else:
        output_data = processed_data
        applied_changes = proposed_changes

    write_json(output_json, output_data)
    write_jsonl(results_jsonl, final_rows)
    write_jsonl(errors_jsonl, final_error_rows)
    write_results_csv(results_csv, final_rows)

    decision_counts = {
        decision: sum(row.get("decyzja") == decision for row in final_rows)
        for decision in sorted(DECISIONS)
    }
    confidence_counts = {
        confidence: sum(row.get("pewnosc") == confidence for row in final_rows)
        for confidence in sorted(CONFIDENCE_LEVELS)
    }
    summary = {
        "plik_zrodlowy": str(args.input),
        "plik_wynikowy": str(output_json),
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "tryb": "tylko_raport" if args.review_only else "zastosowanie_do_kopii",
        "tryb_selekcji": (
            "wskazane_id"
            if args.ids
            else "wszystkie_odsyłacze"
            if args.all_references
            else "kandydaci_z_sygnalem"
        ),
        "selekcja": selection_stats,
        "wybrani": len(selected_tasks),
        "nowe_zapytania": len(pending),
        "nowe_wyniki": completed,
        "nowe_bledy": new_errors,
        "wyniki_lacznie": len(final_rows),
        "bledy_lacznie": len(final_error_rows),
        "decyzje": decision_counts,
        "pewnosc": confidence_counts,
        "stosowanie_sredniej_pewnosci": args.apply_medium_confidence,
        "potencjalne_zmiany": proposed_changes,
        "zastosowane_zmiany": applied_changes,
        "katalog_typow": str(args.settlement_types),
        "slownik_skrotow": str(args.abbreviations),
        "czas_calkowity": format_duration(time.monotonic() - started_at),
        "pliki": {
            "kandydaci_csv": str(candidates_csv),
            "wyniki_jsonl": str(results_jsonl),
            "wyniki_csv": str(results_csv),
            "bledy_jsonl": str(errors_jsonl),
        },
    }
    write_json(summary_json, summary)
    print(output_json)
    print(results_csv)
    print(summary_json)
    print(
        f"Zmiany: rekordy={applied_changes['rekordy']}, "
        f"typ={applied_changes['typ']}, "
        f"typ_punktu_osadniczego="
        f"{applied_changes['typ_punktu_osadniczego']}",
        file=sys.stderr,
    )
    print(f"Czas wykonania: {summary['czas_calkowity']}", file=sys.stderr)
    return 1 if new_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
