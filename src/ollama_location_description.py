"""Dodawanie pola opis_lokalizacji z uzyciem lokalnego modelu Ollama."""

from __future__ import annotations

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
import re
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_MODEL = "gemma4:31b-cloud"
DEFAULT_INPUT = Path("data/example.json")
DEFAULT_OUTPUT_DIR = Path("output")
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_WORKERS = 1
DEFAULT_RETRIES = 5
DEFAULT_RETRY_DELAY = 30.0
PROMPT_VERSION = "location_description_ollama_v2"
TRANSIENT_HTTP_CODES = {408, 425, 429, 500, 502, 503, 504}


class OllamaTransientError(RuntimeError):
    """Blad tymczasowy, po ktorym warto ponowic zapytanie."""


class ModelResponseError(ValueError):
    """Niepoprawna odpowiedz modelu wraz z jej surowa trescia."""

    def __init__(self, message: str, raw_output: str = "") -> None:
        super().__init__(message)
        self.raw_output = raw_output


DESCRIPTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "opis_lokalizacji": {
            "type": "string",
            "description": (
                "Krotki, doslowny lub minimalnie oczyszczony fragment poczatku "
                "hasla zawierajacy informacje lokalizacyjne. Pusty string, "
                "jezeli takich informacji nie ma. Jeżeli w przygotowanym opisie "
                "nie ma nazwy miejscowości, dodaj ją na początku."
            ),
        },
    },
    "required": ["opis_lokalizacji"],
    "additionalProperties": False,
}


def load_env_file(path: Path) -> None:
    if not path.exists():
        return

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
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
        if path in seen:
            continue
        seen.add(path)
        load_env_file(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Uzupelnia rekordy miejscowosci SGKP o pole 'opis_lokalizacji' "
            "wyciagniete z poczatku pola 'text' przez lokalny model Ollama."
        )
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT,
        help="Plik JSON albo katalog z plikami JSON. Domyslnie: data/example.json",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Katalog na pliki wynikowe. Domyslnie: output",
    )
    parser.add_argument(
        "-m",
        "--model",
        default=None,
        help=f"Model Ollama. Domyslnie: {DEFAULT_MODEL} albo OLLAMA_MODEL",
    )
    parser.add_argument(
        "--ollama-url",
        default=None,
        help=f"Adres serwera Ollama. Domyslnie: {DEFAULT_OLLAMA_URL} albo OLLAMA_URL",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
        help="Limit czasu jednego zapytania do Ollamy w sekundach. Domyslnie: 180",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Przelicz wszystkie rekordy i wyzeruj zapisane wyniki czesciowe "
            "oraz bledy dla wskazanego pliku."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maksymalna liczba rekordow do przetworzenia, np. 100 dla probki.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Pauza w sekundach miedzy zapytaniami do Ollamy.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Liczba rownoleglych watkow. Domyslnie: {DEFAULT_WORKERS}.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=DEFAULT_RETRIES,
        help=f"Liczba prob dla jednego rekordu. Domyslnie: {DEFAULT_RETRIES}.",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=DEFAULT_RETRY_DELAY,
        help=(
            "Pauza po bledzie tymczasowym w sekundach. "
            f"Domyslnie: {DEFAULT_RETRY_DELAY:g}."
        ),
    )
    parser.add_argument(
        "--skip-failed",
        action="store_true",
        help=(
            "Nie ponawiaj rekordow, ktore dla tej samej tresci, wersji promptu "
            "i modelu sa juz zapisane w pliku bledow."
        ),
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers musi byc liczba dodatnia")
    if args.retries < 1:
        parser.error("--retries musi byc liczba dodatnia")
    if args.retry_delay < 0:
        parser.error("--retry-delay nie moze byc ujemne")
    if args.sleep < 0:
        parser.error("--sleep nie moze byc ujemne")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit musi byc liczba dodatnia")
    return args


def input_files(input_path: Path) -> list[Path]:
    if input_path.is_dir():
        return sorted(input_path.glob("*.json"))
    return [input_path]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def is_settlement(record: dict[str, Any]) -> bool:
    return "typ_punktu_osadniczego" in record


def iter_target_records(data: Any) -> Iterator[dict[str, Any]]:
    records = data if isinstance(data, list) else [data]
    for record in records:
        if not isinstance(record, dict):
            continue

        if record.get("rodzaj") == "indywidualne" and is_settlement(record):
            yield record
            continue

        if record.get("rodzaj") != "zbiorcze":
            continue

        elements = record.get("elementy")
        if not isinstance(elements, list):
            continue

        for element in elements:
            if (
                isinstance(element, dict)
                and element.get("rodzaj") == "element"
                and is_settlement(element)
            ):
                yield element


def build_prompt(record: dict[str, Any]) -> str:
    name = str(record.get("nazwa", "") or "").strip()
    record_id = str(record.get("ID", "") or "").strip()
    text = str(record.get("text", "") or "").strip()

    return f"""Z poniższego hasła XIX-wiecznego słownika geograficznego wyodrębnij fragment opisujący położenie miejscowości.

Masz wybrać tylko informacje lokalizacyjne z początku hasła: typ miejscowosci, powiat, gmine/parafie, gubernie/wojewodztwo, odległości od innych miejscowości, stacji kolejowych, rzek lub urzędów pocztowych. Zakończ fragment przed informacjami o ludności, domach, historii, wlasności, zabytkach, szkole, cerkwi/kościele albo gospodarce. Jeżeli w przygotowanym opisie nie ma pełnej nazwy miejscowości, dodaj ją na początku na podstawie pola 'nazwa'.

Nie ustalaj wspolczesnej lokalizacji i nie dopisuj faktow spoza tekstu. Zachowaj oryginalne brzmienie na tyle, na ile to mozliwe. Jesli w hasle nie ma informacji lokalizacyjnych, zwroc pusty string.

Nazwa hasla: {name}
ID hasla: {record_id}

Text:
{text}

Odpowiedz wylacznie jako JSON w postaci:
{{"opis_lokalizacji": "..."}}"""


def ollama_generate(
    base_url: str,
    api_key: str | None,
    model: str,
    prompt: str,
    timeout: float,
) -> str:
    url = base_url.rstrip("/")
    if not url.endswith("/api/generate"):
        url += "/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": DESCRIPTION_SCHEMA,
        "options": {
            "temperature": 0,
        },
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    try:
        with urlopen(request, timeout=timeout) as response:
            response_body = response.read().decode("utf-8")
            try:
                response_data = json.loads(response_body)
            except json.JSONDecodeError as exc:
                raise OllamaTransientError(
                    f"Ollama zwrocila odpowiedz inna niz JSON: {response_body}"
                ) from exc
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        message = f"Ollama zwrocila HTTP {exc.code}: {body}"
        if exc.code in TRANSIENT_HTTP_CODES:
            raise OllamaTransientError(message) from exc
        raise RuntimeError(message) from exc
    except (URLError, TimeoutError) as exc:
        raise OllamaTransientError(
            f"Nie mozna polaczyc sie z Ollama ({base_url}): {exc}"
        ) from exc

    if not isinstance(response_data, dict):
        raise OllamaTransientError(
            f"Ollama zwrocila niepoprawna strukture odpowiedzi: {response_data}"
        )
    output = response_data.get("response", "")
    if not isinstance(output, str) or not output.strip():
        raise OllamaTransientError(f"Ollama zwrocila pusta odpowiedz: {response_data}")
    return output.strip()


def parse_model_json(output_text: str) -> dict[str, Any]:
    raw = output_text.strip()
    fenced = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    fenced = re.sub(r"\s*```$", "", fenced).strip()
    candidates = [raw, fenced]

    start = fenced.find("{")
    end = fenced.rfind("}")
    if start != -1 and end >= start:
        candidates.append(fenced[start : end + 1])

    expanded: list[str] = []
    for candidate in candidates:
        if candidate not in expanded:
            expanded.append(candidate)
        if candidate.startswith("{{"):
            unwrapped = candidate[1:].strip()
            if unwrapped not in expanded:
                expanded.append(unwrapped)
        if candidate.endswith("}}"):
            unwrapped = candidate[:-1].strip()
            if unwrapped not in expanded:
                expanded.append(unwrapped)

    last_error: BaseException | None = None
    for candidate in expanded:
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, TypeError) as exc:
            last_error = exc
            try:
                parsed = ast.literal_eval(candidate)
            except (SyntaxError, ValueError) as tolerant_exc:
                last_error = tolerant_exc
                continue
        if isinstance(parsed, dict):
            return parsed
        last_error = TypeError("glowna wartosc JSON nie jest obiektem")

    detail = f": {last_error}" if last_error is not None else ""
    raise ModelResponseError(f"Nie mozna odczytac JSON modelu{detail}", output_text)


def validate_model_result(output_text: str) -> str:
    parsed = parse_model_json(output_text)
    if "opis_lokalizacji" not in parsed:
        raise ModelResponseError(
            "Brak pola opis_lokalizacji w odpowiedzi modelu", output_text
        )
    description = parsed["opis_lokalizacji"]
    if not isinstance(description, str):
        raise ModelResponseError(
            "Pole opis_lokalizacji nie jest napisem", output_text
        )
    return description.strip()


def correction_prompt(prompt: str, error: str, previous_output: str) -> str:
    return f"""{prompt}

Poprzednia odpowiedz byla niepoprawna: {error}
Poprzednia odpowiedz:
{previous_output}

Popraw odpowiedz. Zwroc dokladnie jeden obiekt JSON z jednym polem
opis_lokalizacji, ktorego wartosc jest napisem. Bez Markdownu i komentarza."""


def extract_location_description(
    base_url: str,
    api_key: str | None,
    model: str,
    record: dict[str, Any],
    timeout: float,
    retries: int,
    retry_delay: float,
) -> str:
    text = str(record.get("text", "") or "").strip()
    if not text:
        return ""

    prompt = build_prompt(record)
    label = " ".join(
        value
        for value in (
            get_record_id(record),
            str(record.get("nazwa", "") or "").strip(),
        )
        if value
    )
    last_error: BaseException | None = None

    for attempt in range(1, retries + 1):
        raw_output = ""
        try:
            raw_output = ollama_generate(
                base_url=base_url,
                api_key=api_key,
                model=model,
                prompt=prompt,
                timeout=timeout,
            )
            return validate_model_result(raw_output)
        except OllamaTransientError as exc:
            last_error = exc
            if attempt >= retries:
                break
            print(
                f"Blad tymczasowy {label} ({attempt}/{retries}): {exc}",
                file=sys.stderr,
            )
            if retry_delay > 0:
                time.sleep(retry_delay)
        except ModelResponseError as exc:
            last_error = exc
            if attempt >= retries:
                break
            print(
                f"Bledny wynik modelu {label} ({attempt}/{retries}): {exc}",
                file=sys.stderr,
            )
            prompt = correction_prompt(build_prompt(record), str(exc), exc.raw_output)
            if retry_delay > 0:
                time.sleep(min(retry_delay, 5.0))

    if isinstance(last_error, ModelResponseError):
        raise last_error
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Nieznany blad przetwarzania rekordu {label}")


def get_record_id(record: dict[str, Any]) -> str:
    return str(record.get("ID", "") or "").strip()


def has_description(record: dict[str, Any]) -> bool:
    description = record.get("opis_lokalizacji")
    return isinstance(description, str) and bool(description.strip())


def canonical_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def record_fingerprint(record: dict[str, Any]) -> str:
    relevant_data = {
        "ID": get_record_id(record),
        "nazwa": str(record.get("nazwa", "") or "").strip(),
        "text": str(record.get("text", "") or "").strip(),
    }
    return sha256_text(canonical_json(relevant_data))


def validation_key(input_path: Path, record: dict[str, Any], model: str) -> str:
    key_data = {
        "plik_zrodlowy": input_path.name,
        "record_fingerprint": record_fingerprint(record),
        "model": model,
        "prompt_version": PROMPT_VERSION,
    }
    return sha256_text(canonical_json(key_data))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def partial_output_path(output_path: Path) -> Path:
    return output_path.with_suffix(".partial.jsonl")


def errors_output_path(output_path: Path) -> Path:
    return output_path.with_suffix(".errors.jsonl")


def summary_output_path(output_path: Path) -> Path:
    return output_path.with_suffix(".summary.json")


def load_jsonl_by_key(path: Path) -> tuple[dict[str, dict[str, Any]], int]:
    rows: dict[str, dict[str, Any]] = {}
    ignored = 0
    if not path.exists():
        return rows, ignored

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                print(
                    f"Ostrzezenie: pomijam niepoprawny wiersz {path}:{line_number}",
                    file=sys.stderr,
                )
                ignored += 1
                continue
            if not isinstance(row, dict):
                ignored += 1
                continue
            key = str(row.get("validation_key", "") or "").strip()
            if not key:
                ignored += 1
                continue
            rows[key] = row
    return rows, ignored


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False))
        handle.write("\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")
    tmp_path.replace(path)


def result_row(
    input_path: Path,
    model: str,
    record: dict[str, Any],
    key: str,
    description: str,
) -> dict[str, Any]:
    return {
        "plik_zrodlowy": input_path.name,
        "ID": get_record_id(record),
        "nazwa": str(record.get("nazwa", "") or "").strip(),
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "validation_key": key,
        "record_fingerprint": record_fingerprint(record),
        "opis_lokalizacji": description,
        "czas_utc": utc_now(),
    }


def error_row(
    input_path: Path,
    model: str,
    record: dict[str, Any],
    key: str,
    error: BaseException,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "plik_zrodlowy": input_path.name,
        "ID": get_record_id(record),
        "nazwa": str(record.get("nazwa", "") or "").strip(),
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "validation_key": key,
        "record_fingerprint": record_fingerprint(record),
        "blad": str(error),
        "czas_utc": utc_now(),
    }
    raw_output = getattr(error, "raw_output", "")
    if raw_output:
        row["surowa_odpowiedz_modelu"] = raw_output
    return row


def write_json(path: Path, data: Any) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    tmp_path.replace(path)


def process_record(
    base_url: str,
    api_key: str | None,
    model: str,
    record: dict[str, Any],
    timeout: float,
    sleep_seconds: float,
    retries: int,
    retry_delay: float,
) -> tuple[str, str]:
    description = extract_location_description(
        base_url=base_url,
        api_key=api_key,
        model=model,
        record=record,
        timeout=timeout,
        retries=retries,
        retry_delay=retry_delay,
    )
    if sleep_seconds > 0:
        time.sleep(sleep_seconds)
    return get_record_id(record), description


def process_file(
    base_url: str,
    api_key: str | None,
    model: str,
    input_path: Path,
    output_dir: Path,
    overwrite: bool,
    sleep_seconds: float,
    timeout: float,
    limit: int | None,
    workers: int,
    retries: int,
    retry_delay: float,
    skip_failed: bool,
) -> tuple[Path, int]:
    started_at = time.monotonic()
    data = load_json(input_path)
    targets = list(iter_target_records(data))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{input_path.stem}_opis_lokalizacji_ollama.json"
    partial_path = partial_output_path(output_path)
    errors_path = errors_output_path(output_path)
    summary_path = summary_output_path(output_path)

    if overwrite:
        write_jsonl(partial_path, [])
        write_jsonl(errors_path, [])

    partial_results, ignored_partial = load_jsonl_by_key(partial_path)
    saved_errors, ignored_errors = load_jsonl_by_key(errors_path)
    ignored_rows = ignored_partial + ignored_errors
    if ignored_rows:
        print(
            f"{input_path.name}: pominieto {ignored_rows} starych lub blednych wpisow JSONL",
            file=sys.stderr,
        )

    pending: list[tuple[dict[str, Any], str]] = []
    current_keys: set[str] = set()
    skipped_existing = 0
    restored_from_cache = 0
    skipped_failed_count = 0
    deferred_by_limit = 0

    for record in targets:
        record_id = get_record_id(record)
        if not record_id:
            raise RuntimeError(f"Rekord bez ID: {record}")
        key = validation_key(input_path, record, model)
        current_keys.add(key)

        if has_description(record) and not overwrite:
            partial_results.pop(key, None)
            saved_errors.pop(key, None)
            skipped_existing += 1
            continue

        cached = partial_results.get(key)
        cached_description = cached.get("opis_lokalizacji") if cached else None
        if isinstance(cached_description, str):
            record["opis_lokalizacji"] = cached_description
            saved_errors.pop(key, None)
            restored_from_cache += 1
            continue
        if cached is not None:
            partial_results.pop(key, None)

        if skip_failed and key in saved_errors:
            skipped_failed_count += 1
            continue

        if limit is not None and len(pending) >= limit:
            deferred_by_limit += 1
            continue
        pending.append((record, key))

    completed = 0
    new_errors = 0
    processed = 0
    empty_results = 0
    worker_count = workers

    def save_success(record: dict[str, Any], key: str, description: str) -> None:
        nonlocal completed, processed, empty_results
        record["opis_lokalizacji"] = description
        row = result_row(input_path, model, record, key, description)
        partial_results[key] = row
        saved_errors.pop(key, None)
        append_jsonl(partial_path, row)
        completed += 1
        processed += 1
        if not description:
            empty_results += 1
        print(
            (
                f"Gotowe {processed}/{len(pending)} "
                f"{get_record_id(record)} {record.get('nazwa', '')}"
            ),
            file=sys.stderr,
        )

    def save_error(record: dict[str, Any], key: str, error: BaseException) -> None:
        nonlocal new_errors, processed
        row = error_row(input_path, model, record, key, error)
        saved_errors[key] = row
        append_jsonl(errors_path, row)
        new_errors += 1
        processed += 1
        print(
            (
                f"Blad {processed}/{len(pending)} "
                f"{get_record_id(record)} {record.get('nazwa', '')}: {error}"
            ),
            file=sys.stderr,
        )

    if worker_count == 1:
        for record, key in pending:
            try:
                _, description = process_record(
                    base_url=base_url,
                    api_key=api_key,
                    model=model,
                    record=record,
                    timeout=timeout,
                    sleep_seconds=sleep_seconds,
                    retries=retries,
                    retry_delay=retry_delay,
                )
            except Exception as exc:
                save_error(record, key, exc)
                continue
            save_success(record, key, description)
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures: dict[Any, tuple[dict[str, Any], str]] = {}
            for record, key in pending:
                future = executor.submit(
                    process_record,
                    base_url,
                    api_key,
                    model,
                    record,
                    timeout,
                    sleep_seconds,
                    retries,
                    retry_delay,
                )
                futures[future] = (record, key)

            for future in as_completed(futures):
                record, key = futures[future]
                try:
                    _, description = future.result()
                except Exception as exc:
                    save_error(record, key, exc)
                    continue
                save_success(record, key, description)

    final_results = [row for key, row in partial_results.items() if key in current_keys]
    final_errors = [
        row
        for key, row in saved_errors.items()
        if key in current_keys and key not in partial_results
    ]

    def row_sort_key(row: dict[str, Any]) -> tuple[str, str]:
        return str(row.get("ID", "")), str(row.get("validation_key", ""))

    write_jsonl(partial_path, sorted(final_results, key=row_sort_key))
    write_jsonl(errors_path, sorted(final_errors, key=row_sort_key))
    write_json(output_path, data)

    elapsed = time.monotonic() - started_at
    summary = {
        "plik_zrodlowy": input_path.name,
        "plik_wynikowy": str(output_path),
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "liczba_rekordow_docelowych": len(targets),
        "pominiete_z_wypelnionym_opisem": skipped_existing,
        "przywrocone_z_pamieci_czesciowej": restored_from_cache,
        "pominiete_wczesniejsze_bledy": skipped_failed_count,
        "odlozone_przez_limit": deferred_by_limit,
        "przetworzone_w_tym_uruchomieniu": completed,
        "nowe_bledy": new_errors,
        "wyniki_puste_w_tym_uruchomieniu": empty_results,
        "zapisane_wyniki_czesciowe": len(final_results),
        "zapisane_bledy": len(final_errors),
        "czas_wykonania_sekundy": round(elapsed, 3),
        "czas_wykonania": format_duration(elapsed),
        "czas_utc": utc_now(),
    }
    write_json(summary_path, summary)
    print(
        (
            f"{input_path.name}: przetworzono {completed} rekordow, "
            f"bledy {new_errors}, z cache {restored_from_cache}, "
            f"pominieto wypelnione {skipped_existing}, czas {format_duration(elapsed)}"
        ),
        file=sys.stderr,
    )
    return output_path, new_errors


def format_duration(seconds: float) -> str:
    total_seconds = int(round(seconds))
    minutes, secs = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def main() -> int:
    started_at = time.monotonic()
    args = parse_args()
    load_env_files(args.input)

    files = input_files(args.input)
    missing = [path for path in files if not path.exists()]
    if missing:
        print(f"Nie znaleziono pliku: {missing[0]}", file=sys.stderr)
        return 2

    model = args.model or os.environ.get("OLLAMA_MODEL", DEFAULT_MODEL)
    base_url = args.ollama_url or os.environ.get("OLLAMA_URL", DEFAULT_OLLAMA_URL)
    api_key = os.environ.get("OLLAMA_API_KEY")
    written: list[Path] = []
    total_errors = 0

    for path in files:
        output_path, new_errors = process_file(
            base_url=base_url,
            api_key=api_key,
            model=model,
            input_path=path,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
            sleep_seconds=args.sleep,
            timeout=args.timeout,
            limit=args.limit,
            workers=args.workers,
            retries=args.retries,
            retry_delay=args.retry_delay,
            skip_failed=args.skip_failed,
        )
        written.append(output_path)
        total_errors += new_errors

    for path in written:
        print(path)
    print(
        f"Czas calkowity: {format_duration(time.monotonic() - started_at)}",
        file=sys.stderr,
    )
    return 1 if total_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
