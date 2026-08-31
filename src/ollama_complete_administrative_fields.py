#!/usr/bin/env python3
"""Uzupelnia brakujace pola powiat_ocr i gmina za pomoca Gemmy przez Ollama.

Skrypt przetwarza rekordy indywidualne oraz elementy rekordow zbiorczych.
Istniejacych wartosci nie zmienia. Wynik zapisuje do nowego pliku JSON, a
odpowiedzi modelu, bledy i podsumowanie do osobnych plikow audytowych.
"""

from __future__ import annotations

import argparse
import ast
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
import unicodedata
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_MODEL = "gemma4:31b-cloud"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_OUTPUT_DIR = Path("sgkp_uzupelnienie/ollama")
DEFAULT_WORKERS = 1
DEFAULT_TIMEOUT = 300.0
DEFAULT_RETRIES = 5
DEFAULT_RETRY_DELAY = 30.0
PROMPT_VERSION = "administrative_fields_ollama_v3"
TRANSIENT_HTTP_CODES = {408, 425, 429, 500, 502, 503, 504}
FIELDS = ("powiat_ocr", "gmina")

# Celowo nie dopasowujemy np. slowa "powstaje". Obslugiwane sa formy
# "pow.", "pow ", "powiat...", "gm.", "gm " i odmiany slowa "gmina".
FIELD_CUES = {
    "powiat_ocr": re.compile(r"(?i)(?<!\w)(?:pow(?=\s|[.,:])|powiat\w*)"),
    "gmina": re.compile(
        r"(?i)(?<!\w)(?:gm(?=\s|[.,:])|gmin(?=[.,:])|gmin(?:a|y|ie|ę|ą|ach|ami)\b)"
    ),
}

GMINA_MARKER_RE = re.compile(r"(?i)(?<!\w)(?:gm(?=\s|[.,:])|gmin\w*)")
GMINA_NON_MEMBERSHIP_RE = re.compile(
    r"(?ix)"
    r"(?:\b(?:zarz\w*|zarząd\w*|sąd\w*|urząd\w*|szk\w*)\b"
    r".{0,60}\b(?:gm(?=\s|[.,:])|gmin\w*)\b)"
    r"|(?:\b(?:gm(?=\s|[.,:])|gmin\w*)\b"
    r".{0,60}\b(?:zarz\w*|zarząd\w*|sąd\w*|urząd\w*|szk\w*)\b)"
    r"|(?:\bgmin\w*\b\s+(?:składa\w*|licz\w*|posiada\w*|obejmuje\w*))"
    r"|(?:\b(?:obszar\w*|ludność\w*|ludn\w*)\b\s+\bgmin\w*\b)"
    r"|(?:\b(?:przez|dla)\s+\bgmin\w*\b)"
)
GMINA_SAME_NAME_RE = re.compile(
    r"(?ix)"
    r"(?:\b(?:gm(?=\s|[.,:])|gmin\w*)[.,]?\s*"
    r"(?:t[.]?\s*n[.]?|tej\s+nazwy|tegoż\s+nazwania)\b)"
    r"|(?:\b(?:wś|wieś|mko|miasto|osada|folwark|dominium)\b"
    r"[^.;]{0,20}\b(?:i|oraz)\s+(?:gm(?=\s|[.,:])|gmina)\b)"
    r"|(?:\b(?:gm(?=\s|[.,:])|gmina)\b[.,]?\s+(?:i|oraz)\s+"
    r"(?:wś|wieś|mko|miasto|osada|folwark|dominium)\b)"
    r"|(?:\bgm\b[.,]?\s+(?:wiejska|miejska)\b)"
)


class OllamaTransientError(RuntimeError):
    """Blad tymczasowy, po ktorym warto ponowic zapytanie."""


class ModelResponseError(ValueError):
    """Niepoprawna odpowiedz modelu wraz z surowa trescia."""

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


def has_value(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    return value.strip().casefold() not in {"", "/", "null", "none", "brak"}


def task_label(task: dict[str, Any]) -> str:
    name = " ".join(str(task.get("nazwa", "") or "").split())
    return f"{task['ID']} {name}".strip()


def field_has_cue(text: str, field: str) -> bool:
    return bool(FIELD_CUES[field].search(text))


def text_scoped_to_element(elements: list[Any], element_index: int) -> str:
    """Usuwa doklejona kopie nastepnego elementu z kontekstu zadania.

    Nie zmienia danych zrodlowych. Zabezpiecza ekstrakcje, gdy po naprawie
    hasla zbiorczego tekst jednego elementu omylkowo zawiera na koncu pelny
    tekst kolejnego elementu.
    """

    current = elements[element_index]
    if not isinstance(current, dict):
        return ""
    text = str(current.get("text", "") or "").strip()
    cut_positions: list[int] = []
    for later in elements[element_index + 1 :]:
        if not isinstance(later, dict):
            continue
        later_text = str(later.get("text", "") or "").strip()
        if len(later_text) < 10:
            continue
        position = text.find(later_text)
        if position > 0:
            cut_positions.append(position)
    if cut_positions:
        return text[: min(cut_positions)].rstrip()
    return text


def collect_tasks(data: Any, all_missing: bool) -> list[dict[str, Any]]:
    """Buduje zadania dla rekordow indywidualnych i elementow zbiorczych."""

    if not isinstance(data, list):
        raise RuntimeError("Plik wejsciowy powinien zawierac liste rekordow")
    tasks: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    def add_task(
        target: dict[str, Any],
        record_index: int,
        element_index: int | None,
        parent: dict[str, Any] | None,
        scoped_text: str | None = None,
    ) -> None:
        missing_fields = [field for field in FIELDS if not has_value(target.get(field))]
        if not missing_fields:
            return
        text = (
            scoped_text
            if scoped_text is not None
            else str(target.get("text", "") or "").strip()
        )
        if not text:
            return
        cue_fields = [field for field in missing_fields if field_has_cue(text, field)]
        if not all_missing and not cue_fields:
            return
        target_id = str(target.get("ID", "") or "").strip()
        if not target_id:
            raise RuntimeError(
                f"Brak ID dla rekordu {record_index}, elementu {element_index}"
            )
        if target_id in seen_ids:
            raise RuntimeError(f"Powtorzone ID zadania: {target_id}")
        seen_ids.add(target_id)
        tasks.append(
            {
                "ID": target_id,
                "nazwa": target.get("nazwa"),
                "rodzaj_celu": "element" if element_index is not None else "indywidualne",
                "parent_ID": (
                    str(parent.get("ID", "") or "").strip() if parent else None
                ),
                "record_index": record_index,
                "element_index": element_index,
                "text": text,
                "brakujace_pola": missing_fields,
                "pola_z_sygnalem_w_tekscie": cue_fields,
                "istniejace_wartosci": {
                    field: target.get(field) for field in FIELDS if has_value(target.get(field))
                },
            }
        )

    for record_index, record in enumerate(data):
        if not isinstance(record, dict):
            continue
        if record.get("rodzaj") == "indywidualne":
            add_task(record, record_index, None, None)
        elif record.get("rodzaj") == "zbiorcze":
            elements = record.get("elementy", [])
            if not isinstance(elements, list):
                continue
            for element_index, element in enumerate(elements):
                if isinstance(element, dict):
                    add_task(
                        element,
                        record_index,
                        element_index,
                        record,
                        text_scoped_to_element(elements, element_index),
                    )
    return tasks


def task_fingerprint(task: dict[str, Any]) -> str:
    payload = {
        "ID": task["ID"],
        "nazwa": task.get("nazwa"),
        "text": task["text"],
        "brakujace_pola": task["brakujace_pola"],
        "istniejace_wartosci": task["istniejace_wartosci"],
    }
    return sha256_text(canonical_json(payload))


def validation_key(source_name: str, task: dict[str, Any], model: str) -> str:
    payload = {
        "source": source_name,
        "task_fingerprint": task_fingerprint(task),
        "model": model,
        "prompt_version": PROMPT_VERSION,
    }
    return sha256_text(canonical_json(payload))


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
                    "Wydobywasz tylko dane jawnie obecne w dostarczonym "
                    "tekscie. Odpowiadasz wylacznie poprawnym JSON-em."
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


def build_prompt(task: dict[str, Any]) -> str:
    fields = ", ".join(task["brakujace_pola"])
    existing = task["istniejace_wartosci"] or "brak"
    return f"""Uzupelnij wylacznie brakujace dane administracyjne opisywanego hasla lub podhasla SGKP.

ID: {task['ID']}
Nazwa: {task.get('nazwa')}
Rodzaj: {task['rodzaj_celu']}
Brakujace pola: {fields}
Istniejace dane administracyjne, ktorych nie wolno zmieniac: {json.dumps(existing, ensure_ascii=False)}

TEKST:
{task['text']}

REGULY:
1. Korzystaj tylko z powyzszego tekstu. Nie uzywaj wiedzy zewnetrznej i nie zgaduj.
2. `powiat_ocr` to nazwa powiatu, do ktorego nalezy glowny obiekt opisywany przez ten tekst. Zwykle wystepuje po `pow.` lub slowie `powiat`. Zachowaj historyczna pisownie tekstu i pomin samo slowo/skrot `powiat`.
3. `gmina` to nazwa gminy, do ktorej tekst JAWNIE przypisuje glowny obiekt. Sam fakt, ze miejscowosc ma zarzad, urzad, sad lub szkole gminna, NIE podaje nazwy gminy i nie pozwala uzupelnic tego pola nazwa hasla.
4. Skrót `g.` NIE oznacza gminy. W tekstach SGKP oznacza zwykle gubernię. Także `gub.` oznacza gubernię. Nigdy nie uzupełniaj pola `gmina` na podstawie `g.` ani `gub.`.
5. Gdy tekst nazywa opisywany obiekt miastem powiatowym (`m. pow.`) albo samym powiatem, nazwa powiatu jest nazwa hasla.
6. Nazwe hasla wolno wpisac jako `gmina` tylko wtedy, gdy tekst jawnie nazywa opisywany obiekt gmina, np. `wś i gm.`, `gm. t. n.` albo `gmina tej nazwy`. Nie stosuj tej reguly do zwrotow `z zarzadem gminnym`, `posiada zarzad gm.`, `zarz. gm. t. n.` ani podobnych informacji o siedzibie administracji.
7. Nie uznawaj za przynaleznosc administracyjna wzmianek o sasiednich miejscach, odleglosci od miasta powiatowego, zarzadzie, urzedzie, sadzie lub szkole gminnej, statystyce calej gminy lub powiatu ani wykazu miejscowosci nalezacych do gminy. Zwroty `gmina sklada sie`, `gmina liczy`, `obszar gminy` i `zarzad gminny` nie sa dowodem nazwy gminy glownego obiektu.
8. Jezeli tekst opisuje kilka obiektow nalezacych do roznych powiatow lub gmin i nie da sie wskazac jednej wartosci dla glownego obiektu, zwroc `niejednoznaczne`.
9. `dowod` ma byc krotkim, doslownym cytatem z TEKSTU. Dla pola `gmina` dowod powinien zawierac jawna nazwe gminy i oznaczenie `gm.`/`gmina`; wyjatkiem jest jednoznaczny zapis `gm. t. n.` lub `gmina tej nazwy`. Nie umieszczaj w dowodzie parafrazy.
10. Dla pola, ktorego nie ma jawnie w tekscie, zwroc `brak_w_tekscie`. Nie uzupelniaj go na podstawie drugiego pola.
11. Zwroc oba brakujace pola wymienione wyzej. Nie zwracaj ani nie zmieniaj innych danych.

PRZYKLAD NEGATYWNY: dla tekstu `Dernowicze, wś z zarządem gminnym. Gmina składa się z 37 wsi` zwroc dla `gmina` status `brak_w_tekscie`. Tekst nie zawiera informacji `gm. Dernowicze`; nie wolno jej dopowiadac z nazwy hasla.

Dozwolone statusy: `znaleziono`, `brak_w_tekscie`, `niejednoznaczne`.

Zwroc wylacznie taki obiekt JSON:
{{
  "powiat_ocr": {{
    "status": "znaleziono",
    "wartosc": "woliński",
    "dowod": "pow. woliński"
  }},
  "gmina": {{
    "status": "znaleziono",
    "wartosc": "Pastwiska",
    "dowod": "gm. Pastwiska"
  }},
  "uwagi": ""
}}

Jesli jedno z pol nie jest wymienione w `Brakujace pola`, mozesz je pominac albo zwrocic jako null."""


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
    return common >= max(4, shortest - 2)


def value_is_present_in_evidence(value: str, evidence: str) -> bool:
    value_words = word_tokens(value)
    evidence_words = word_tokens(evidence)
    if not value_words or not evidence_words:
        return False
    return all(
        any(
            words_roughly_match(value_word, evidence_word)
            for evidence_word in evidence_words
        )
        for value_word in value_words
    )


def validate_explicit_gmina(value: str, evidence: str, task: dict[str, Any]) -> None:
    """Odrzuca wnioski o gminie oparte tylko na posrednich przeslankach."""

    if not GMINA_MARKER_RE.search(evidence):
        raise ValueError("Dowod pola gmina nie zawiera oznaczenia gm./gmina")
    if GMINA_NON_MEMBERSHIP_RE.search(evidence):
        raise ValueError(
            "Dowod pola gmina opisuje zarzad, urzad, sad, szkole lub "
            "statystyke gminy, a nie jawna przynaleznosc miejscowosci"
        )
    if value_is_present_in_evidence(value, evidence):
        return

    task_name = str(task.get("nazwa", "") or "").strip()
    same_as_task_name = (
        normalized_text(value) == normalized_text(task_name)
        if task_name
        else False
    )
    if same_as_task_name and GMINA_SAME_NAME_RE.search(evidence):
        return
    raise ValueError(
        "Dowod pola gmina nie zawiera jawnej nazwy gminy ani "
        "jednoznacznego zapisu, ze gmina jest tej samej nazwy"
    )


def normalize_field_result(
    raw: Any,
    field: str,
    task: dict[str, Any],
) -> dict[str, Any]:
    source_text = task["text"]
    if raw is None:
        return {
            "status": "brak_w_tekscie",
            "wartosc": None,
            "dowod": "",
        }
    if isinstance(raw, str):
        raw = {"status": "znaleziono", "wartosc": raw, "dowod": raw}
    if not isinstance(raw, dict):
        raise ValueError(f"Pole {field} nie jest obiektem")

    status = str(raw.get("status", "") or "").strip().casefold()
    aliases = {
        "brak": "brak_w_tekscie",
        "nie znaleziono": "brak_w_tekscie",
        "nie_znaleziono": "brak_w_tekscie",
        "niejednoznaczny": "niejednoznaczne",
        "niejednoznaczna": "niejednoznaczne",
    }
    status = aliases.get(status, status)
    if status not in {"znaleziono", "brak_w_tekscie", "niejednoznaczne"}:
        raise ValueError(f"Niepoprawny status pola {field}: {status!r}")

    value = str(raw.get("wartosc", "") or "").strip()
    evidence = str(raw.get("dowod", "") or "").strip()
    if status == "znaleziono":
        if not has_value(value):
            raise ValueError(f"Brak wartosci dla pola {field}")
        if not evidence:
            raise ValueError(f"Brak dowodu dla pola {field}")
        if normalized_text(evidence) not in normalized_text(source_text):
            raise ValueError(f"Dowod dla pola {field} nie wystepuje doslownie w tekscie")
        if field == "gmina":
            validate_explicit_gmina(value, evidence, task)
        return {"status": status, "wartosc": value, "dowod": evidence}
    return {"status": status, "wartosc": None, "dowod": evidence}


def normalize_response(
    parsed: dict[str, Any],
    task: dict[str, Any],
) -> dict[str, Any]:
    results: dict[str, dict[str, Any]] = {}
    for field in task["brakujace_pola"]:
        results[field] = normalize_field_result(parsed.get(field), field, task)
    warnings = parsed.get("uwagi", "")
    if isinstance(warnings, list):
        warning_text = " | ".join(str(item).strip() for item in warnings if str(item).strip())
    else:
        warning_text = str(warnings or "").strip()
    return {"wyniki_pol": results, "uwagi": warning_text}


def extract_with_retries(
    base_url: str,
    api_key: str | None,
    model: str,
    task: dict[str, Any],
    timeout: float,
    retries: int,
    retry_delay: float,
) -> dict[str, Any]:
    prompt = build_prompt(task)
    current_prompt = prompt
    attempts = max(1, retries)
    for attempt in range(1, attempts + 1):
        output = ""
        try:
            output = ollama_chat(base_url, api_key, model, current_prompt, timeout)
            return normalize_response(parse_model_json(output), task)
        except OllamaTransientError as exc:
            if attempt == attempts:
                raise
            print(
                f"{task_label(task)}: blad tymczasowy ({attempt}/{attempts}): "
                f"{short_error(exc)}",
                file=sys.stderr,
            )
            time.sleep(max(0.0, retry_delay))
        except ValueError as exc:
            if attempt == attempts:
                raw_output = getattr(exc, "raw_output", output)
                raise ModelResponseError(short_error(exc), raw_output) from exc
            print(
                f"{task_label(task)}: bledny wynik modelu ({attempt}/{attempts}): "
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
    timeout: float,
    retries: int,
    retry_delay: float,
) -> dict[str, Any]:
    result = extract_with_retries(
        base_url,
        api_key,
        model,
        task,
        timeout,
        retries,
        retry_delay,
    )
    additions = {
        field: field_result["wartosc"]
        for field, field_result in result["wyniki_pol"].items()
        if field_result["status"] == "znaleziono"
    }
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
        "validation_key": validation_key(source_path.name, task, model),
        "task_fingerprint": task_fingerprint(task),
        "brakujace_pola": task["brakujace_pola"],
        "pola_z_sygnalem_w_tekscie": task["pola_z_sygnalem_w_tekscie"],
        **result,
        "uzupelnienia": additions,
        "czas_utc": datetime.now(timezone.utc).isoformat(),
    }


def make_error_row(
    source_path: Path,
    task: dict[str, Any],
    model: str,
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
        "validation_key": validation_key(source_path.name, task, model),
        "task_fingerprint": task_fingerprint(task),
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


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False))
        handle.write("\n")


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


def write_candidates_csv(path: Path, tasks: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    columns = [
        "ID",
        "nazwa",
        "rodzaj_celu",
        "parent_ID",
        "brakujace_pola",
        "pola_z_sygnalem_w_tekscie",
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
                    "brakujace_pola": " | ".join(task["brakujace_pola"]),
                    "pola_z_sygnalem_w_tekscie": " | ".join(
                        task["pola_z_sygnalem_w_tekscie"]
                    ),
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
        "powiat_ocr_status",
        "powiat_ocr_wartosc",
        "powiat_ocr_dowod",
        "gmina_status",
        "gmina_wartosc",
        "gmina_dowod",
        "uwagi",
    ]
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            field_results = row.get("wyniki_pol", {})
            csv_row: dict[str, Any] = {
                "ID": row.get("ID"),
                "nazwa": row.get("nazwa"),
                "rodzaj_celu": row.get("rodzaj_celu"),
                "parent_ID": row.get("parent_ID"),
                "uwagi": row.get("uwagi"),
            }
            for field in FIELDS:
                result = field_results.get(field, {})
                csv_row[f"{field}_status"] = result.get("status")
                csv_row[f"{field}_wartosc"] = result.get("wartosc")
                csv_row[f"{field}_dowod"] = result.get("dowod")
            writer.writerow(csv_row)
    temporary.replace(path)


def resolve_target(data: list[Any], row: dict[str, Any]) -> dict[str, Any]:
    record_index = int(row["record_index"])
    record = data[record_index]
    if not isinstance(record, dict):
        raise RuntimeError(f"Rekord {record_index} nie jest obiektem")
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


def apply_results(data: Any, rows: list[dict[str, Any]]) -> dict[str, int]:
    if not isinstance(data, list):
        raise RuntimeError("Plik wejsciowy powinien zawierac liste rekordow")
    counts = {field: 0 for field in FIELDS}
    for row in rows:
        target = resolve_target(data, row)
        for field, value in row.get("uzupelnienia", {}).items():
            if field in FIELDS and not has_value(target.get(field)) and has_value(value):
                target[field] = value
                counts[field] += 1
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Uzupelnia brakujace pola powiat_ocr i gmina w rekordach "
            "indywidualnych oraz elementach zbiorczych przez Gemme/Ollama."
        )
    )
    parser.add_argument("input", type=Path, help="Zrodlowy plik sgkp_XX.json")
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
        "--all-missing",
        action="store_true",
        help=(
            "Analizuj wszystkie rekordy z brakujacym polem, nawet bez slow "
            "powiat/gmina w tekscie. Moze znacznie zwiekszyc liczbe zapytan."
        ),
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
    return args


def main() -> int:
    started_at = time.monotonic()
    args = parse_args()
    if not args.input.is_file():
        print(f"Nie znaleziono pliku: {args.input}", file=sys.stderr)
        return 2
    try:
        data = load_json(args.input)
        tasks = collect_tasks(data, args.all_missing)
    except (OSError, json.JSONDecodeError, RuntimeError) as exc:
        print(f"Nie mozna przygotowac danych: {short_error(exc)}", file=sys.stderr)
        return 2

    if args.ids:
        requested_ids = {
            item.strip()
            for value in args.ids
            for item in value.split(",")
            if item.strip()
        }
        available_ids = {task["ID"] for task in tasks}
        missing_ids = sorted(requested_ids - available_ids)
        if missing_ids:
            print(
                "Wskazane ID nie sa kandydatami: " + ", ".join(missing_ids),
                file=sys.stderr,
            )
            return 2
        tasks = [task for task in tasks if task["ID"] in requested_ids]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_dir / f"{args.input.stem}_administrative_completion"
    candidates_csv = prefix.with_suffix(".candidates.csv")
    results_jsonl = prefix.with_suffix(".results.jsonl")
    results_csv = prefix.with_suffix(".results.csv")
    errors_jsonl = prefix.with_suffix(".errors.jsonl")
    summary_json = prefix.with_suffix(".summary.json")
    output_json = args.output_dir / f"{args.input.stem}_admin_completed.json"
    write_candidates_csv(candidates_csv, tasks)
    print(f"Kandydaci={len(tasks)}", file=sys.stderr)
    if args.scan_only:
        print(candidates_csv)
        return 0

    load_env_files(args.input)
    model = args.model or os.environ.get("OLLAMA_MODEL", DEFAULT_MODEL)
    base_url = args.ollama_url or os.environ.get("OLLAMA_URL", DEFAULT_OLLAMA_URL)
    api_key = os.environ.get("OLLAMA_API_KEY")

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

    pending: list[dict[str, Any]] = []
    for task in tasks:
        key = validation_key(args.input.name, task, model)
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

    if args.workers == 1:
        for processed, task in enumerate(pending, start=1):
            try:
                row = make_result_row(
                    args.input,
                    task,
                    base_url,
                    api_key,
                    model,
                    args.timeout,
                    args.retries,
                    args.retry_delay,
                )
            except Exception as exc:
                new_errors += 1
                error_row = make_error_row(args.input, task, model, exc)
                existing_errors[error_row["validation_key"]] = error_row
                print(
                    f"[{processed}/{len(pending)}] Blad {task_label(task)}: "
                    f"{short_error(exc)}",
                    file=sys.stderr,
                )
            else:
                handle_success(row)
                additions = ", ".join(
                    f"{field}={value}" for field, value in row["uzupelnienia"].items()
                ) or "brak danych do dodania"
                print(
                    f"[{processed}/{len(pending)}] Gotowe {task_label(task)}: "
                    f"{additions}",
                    file=sys.stderr,
                )
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    make_result_row,
                    args.input,
                    task,
                    base_url,
                    api_key,
                    model,
                    args.timeout,
                    args.retries,
                    args.retry_delay,
                ): task
                for task in pending
            }
            for processed, future in enumerate(as_completed(futures), start=1):
                task = futures[future]
                try:
                    row = future.result()
                except Exception as exc:
                    new_errors += 1
                    error_row = make_error_row(args.input, task, model, exc)
                    existing_errors[error_row["validation_key"]] = error_row
                    print(
                        f"[{processed}/{len(pending)}] Blad {task_label(task)}: "
                        f"{short_error(exc)}",
                        file=sys.stderr,
                    )
                else:
                    handle_success(row)
                    additions = ", ".join(
                        f"{field}={value}"
                        for field, value in row["uzupelnienia"].items()
                    ) or "brak danych do dodania"
                    print(
                        f"[{processed}/{len(pending)}] Gotowe {task_label(task)}: "
                        f"{additions}",
                        file=sys.stderr,
                    )

    current_keys = {
        validation_key(args.input.name, task, model) for task in tasks
    }
    final_rows = [existing[key] for key in current_keys if key in existing]
    final_rows.sort(key=lambda row: str(row.get("ID", "")))
    final_error_rows = [
        existing_errors[key] for key in current_keys if key in existing_errors
    ]
    final_error_rows.sort(key=lambda row: str(row.get("ID", "")))
    write_jsonl(results_jsonl, final_rows)
    write_jsonl(errors_jsonl, final_error_rows)
    write_results_csv(results_csv, final_rows)

    additions = apply_results(data, final_rows)
    write_json(output_json, data)
    status_counts = {
        field: {
            status: sum(
                row.get("wyniki_pol", {}).get(field, {}).get("status") == status
                for row in final_rows
            )
            for status in ("znaleziono", "brak_w_tekscie", "niejednoznaczne")
        }
        for field in FIELDS
    }
    summary = {
        "plik_zrodlowy": str(args.input),
        "plik_wynikowy": str(output_json),
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "tryb_kandydatow": "wszystkie_braki" if args.all_missing else "sygnal_w_tekscie",
        "kandydaci": len(tasks),
        "nowe_zapytania": len(pending),
        "nowe_wyniki": completed,
        "nowe_bledy": new_errors,
        "wyniki_lacznie": len(final_rows),
        "bledy_lacznie": len(final_error_rows),
        "uzupelnione_pola": additions,
        "statusy": status_counts,
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
    print(f"Czas wykonania: {summary['czas_calkowity']}", file=sys.stderr)
    return 1 if new_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
