"""Weryfikacja pola l_mk_statystyka za pomoca Gemmy 4 w Ollama Cloud.

Skrypt czyta rekordy SGKP bez modyfikowania plikow zrodlowych. Wyniki audytu
zapisuje jako JSONL i CSV w osobnym katalogu.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
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
DEFAULT_TIMEOUT = 300.0
DEFAULT_RETRIES = 5
DEFAULT_RETRY_DELAY = 60.0
DEFAULT_WORKERS = 1
DEFAULT_INPUT = Path("sgkp_przekazane")
DEFAULT_OUTPUT_DIR = Path("sgkp_weryfikacja/ollama")
DEFAULT_MAX_CONTEXT_CHARS = 30_000
DEFAULT_SNIPPET_RADIUS = 700
PROMPT_VERSION = "l_mk_ollama_v3"
TRANSIENT_HTTP_CODES = {408, 425, 429, 500, 502, 503, 504}

ALLOWED_ITEM_STATUSES = {
    "zgodne_doslowne",
    "zgodne_po_obliczeniu",
    "zgodne_z_zastrzezeniem",
    "zgodny_brak_danych",
    "niezgodne",
    "brak_w_tekscie",
    "niejednoznaczne_ocr",
    "poza_zakresem",
}

POPULATION_ANCHOR_RE = re.compile(
    r"(?:\bmk\.?|\bmieszk\w*|\bludn\w*|\bludność\b|\bludu\b|"
    r"\bosób\b|\bkat\.?|\bewang\w*|\bizrael\w*|\bżyd\w*)",
    flags=re.IGNORECASE,
)

class OllamaTransientError(RuntimeError):
    """Blad tymczasowy, po ktorym warto ponowic zapytanie."""


class ModelResponseError(ValueError):
    """Blad odpowiedzi modelu wraz z jej surowa trescia."""

    def __init__(self, message: str, raw_output: str = "") -> None:
        super().__init__(message)
        self.raw_output = raw_output


def load_env_file(path: Path) -> None:
    """Wczytuje prosty plik .env bez zewnetrznej biblioteki dotenv."""

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
    resolved_input = input_path.resolve()
    input_directory = resolved_input if resolved_input.is_dir() else resolved_input.parent
    candidates = [
        Path.cwd() / ".env",
        Path(__file__).resolve().parent / ".env",
        input_directory / ".env",
    ]
    seen: set[Path] = set()
    for path in candidates:
        if path not in seen:
            seen.add(path)
            load_env_file(path)


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
    """Iteruje po haslach indywidualnych i elementach hasel zbiorczych."""

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
    """Wywoluje lokalne API Ollamy lub bezposrednie API ollama.com."""

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Odpowiadasz wyłącznie poprawnym JSON-em. Nie dodawaj "
                    "markdown, komentarzy ani tekstu poza obiektem JSON."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {"temperature": 0},
    }
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    url = ollama_chat_url(base_url)
    request = Request(
        url,
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
            "Nie mozna polaczyc sie z Ollama. Sprawdz, czy dziala `ollama serve`, "
            f"czy wykonano `ollama signin` i czy adres jest poprawny: {base_url}"
        ) from exc

    try:
        response_data = json.loads(response_text)
    except json.JSONDecodeError as exc:
        excerpt = " ".join(response_text.split())[:500]
        raise OllamaTransientError(
            f"Ollama zwrocila niepoprawny JSON odpowiedzi HTTP: {excerpt}"
        ) from exc
    if not isinstance(response_data, dict):
        raise OllamaTransientError("Odpowiedz Ollamy nie jest obiektem JSON")
    if response_data.get("error"):
        raise RuntimeError(f"Ollama zwrocila blad: {response_data['error']}")
    message = response_data.get("message")
    if not isinstance(message, dict):
        raise OllamaTransientError(f"Ollama zwrocila odpowiedz bez message: {response_data}")
    content = message.get("content", "")
    if not isinstance(content, str) or not content.strip():
        raise OllamaTransientError(f"Ollama zwrocila pusta odpowiedz: {response_data}")
    return content.strip()


def short_error(error: Exception) -> str:
    message = str(error).strip() or error.__class__.__name__
    message = " ".join(message.split())
    return message if len(message) <= 500 else message[:497] + "..."


def format_duration(seconds: float) -> str:
    total_seconds = int(round(seconds))
    minutes, secs = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Weryfikuje l_mk_statystyka na podstawie pola text przy uzyciu "
            "Gemmy 4 31B w Ollama Cloud. Nie modyfikuje plikow zrodlowych."
        )
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT,
        help=(
            "Plik JSON albo katalog z plikami JSON. Domyslnie: "
            f"{DEFAULT_INPUT}"
        ),
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Katalog wynikowy. Domyslnie: {DEFAULT_OUTPUT_DIR}",
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
        help=(
            "Adres Ollamy. Domyslnie: "
            f"{DEFAULT_OLLAMA_URL} albo OLLAMA_URL"
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Liczba rownoleglych zapytan. Domyslnie: {DEFAULT_WORKERS}",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maksymalna liczba nowych zapytan w calym uruchomieniu.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help=f"Timeout jednego zapytania. Domyslnie: {DEFAULT_TIMEOUT} s",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=DEFAULT_RETRIES,
        help=f"Liczba prob dla rekordu. Domyslnie: {DEFAULT_RETRIES}",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=DEFAULT_RETRY_DELAY,
        help=f"Pauza po bledzie tymczasowym. Domyslnie: {DEFAULT_RETRY_DELAY} s",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Dodatkowa pauza po udanym zapytaniu.",
    )
    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=DEFAULT_MAX_CONTEXT_CHARS,
        help=(
            "Dla dluzszych tekstow wysyla wybrane fragmenty zwiazane z "
            "ludnoscia. 0 oznacza zawsze pelny tekst. Domyslnie: "
            f"{DEFAULT_MAX_CONTEXT_CHARS}"
        ),
    )
    parser.add_argument(
        "--snippet-radius",
        type=int,
        default=DEFAULT_SNIPPET_RADIUS,
        help=(
            "Promien fragmentu wokol liczby lub okreslenia ludnosci. "
            f"Domyslnie: {DEFAULT_SNIPPET_RADIUS} znakow"
        ),
    )
    parser.add_argument(
        "--include-empty",
        action="store_true",
        help=(
            "Sprawdzaj rowniez rekordy z pustym l_mk_statystyka pod katem "
            "pominietych danych. Domyslnie sa pomijane."
        ),
    )
    parser.add_argument(
        "--skip-failed",
        action="store_true",
        help=(
            "Pomin rekordy obecne w nierozwiazanym logu *.errors.jsonl. "
            "Bez tej opcji skrypt probuje je ponownie."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Usun dotychczasowe wyniki czastkowe dla przetwarzanych plikow.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Tylko policz rekordy; nie lacz sie z API i niczego nie zapisuj.",
    )
    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        parser.error("--limit musi byc dodatni")
    if args.workers < 1:
        parser.error("--workers musi byc dodatnie")
    if args.retries < 1:
        parser.error("--retries musi byc dodatnie")
    if args.max_context_chars < 0:
        parser.error("--max-context-chars nie moze byc ujemne")
    if args.snippet_radius < 100:
        parser.error("--snippet-radius musi wynosic co najmniej 100")
    return args


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def get_record_id(record: dict[str, Any]) -> str:
    return str(record.get("ID", "") or "").strip()


def record_label(record: dict[str, Any]) -> str:
    """Zwraca czytelna etykiete rekordu do komunikatow postepu."""

    record_id = get_record_id(record)
    name = " ".join(str(record.get("nazwa", "") or "").split())
    return f"{record_id} {name}".strip()


def has_population_statistics(record: dict[str, Any], include_empty: bool) -> bool:
    if include_empty:
        return True
    value = record.get("l_mk_statystyka")
    return isinstance(value, list) and bool(value)


def build_assertions(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Splaszcza l_mk_statystyka, zachowujac indeksy elementow zrodla."""

    statistics = record.get("l_mk_statystyka")
    if not isinstance(statistics, list):
        return []

    assertions: list[dict[str, Any]] = []
    for statistic_index, statistic in enumerate(statistics, start=1):
        if not isinstance(statistic, dict):
            assertions.append(
                {
                    "indeks_statystyki": statistic_index,
                    "indeks_liczby": 0,
                    "dotyczy": None,
                    "data": None,
                    "liczba_json": None,
                    "blad_struktury": "Element l_mk_statystyka nie jest obiektem.",
                }
            )
            continue

        subject = statistic.get("dotyczy")
        numbers = statistic.get("liczba")
        if not isinstance(numbers, list) or not numbers:
            assertions.append(
                {
                    "indeks_statystyki": statistic_index,
                    "indeks_liczby": 0,
                    "dotyczy": subject,
                    "data": None,
                    "liczba_json": None,
                    "blad_struktury": (
                        None
                        if numbers is None
                        else "Pole liczba nie jest niepusta lista."
                    ),
                }
            )
            continue

        for number_index, number in enumerate(numbers, start=1):
            if isinstance(number, dict):
                date = number.get("data")
                value = number.get("liczba")
                structure_error = None
            else:
                date = None
                value = None
                structure_error = "Element pola liczba nie jest obiektem."
            assertions.append(
                {
                    "indeks_statystyki": statistic_index,
                    "indeks_liczby": number_index,
                    "dotyczy": subject,
                    "data": date,
                    "liczba_json": value,
                    "blad_struktury": structure_error,
                }
            )
    return assertions


def validation_key(
    source_name: str,
    record: dict[str, Any],
    model: str,
) -> str:
    payload = {
        "provider": "ollama",
        "source": source_name,
        "ID": get_record_id(record),
        "text": record.get("text"),
        "l_mk_statystyka": record.get("l_mk_statystyka"),
        "model": model,
        "prompt_version": PROMPT_VERSION,
    }
    return sha256_text(canonical_json(payload))


def merge_windows(windows: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not windows:
        return []
    merged: list[list[int]] = []
    for start, end in sorted(windows):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def statistic_number_tokens(assertions: list[dict[str, Any]]) -> set[str]:
    tokens: set[str] = set()
    for assertion in assertions:
        value = assertion.get("liczba_json")
        if value is None:
            continue
        for match in re.finditer(r"\d+", str(value)):
            tokens.add(match.group(0))
    return tokens


def prepare_text_context(
    text: str,
    assertions: list[dict[str, Any]],
    max_chars: int,
    radius: int,
) -> tuple[str, str]:
    """Zwraca pelny tekst lub zlaczone fragmenty dla bardzo dlugiego hasla."""

    if max_chars <= 0 or len(text) <= max_chars:
        return text, "pelny"

    windows: list[tuple[int, int]] = [(0, min(len(text), 1_000))]
    anchors = list(POPULATION_ANCHOR_RE.finditer(text))
    number_tokens = statistic_number_tokens(assertions)

    for match in anchors:
        windows.append(
            (max(0, match.start() - radius), min(len(text), match.end() + radius))
        )
    for token in number_tokens:
        token_re = re.compile(rf"(?<!\d){re.escape(token)}(?!\d)")
        for match in token_re.finditer(text):
            windows.append(
                (max(0, match.start() - radius), min(len(text), match.end() + radius))
            )

    pieces: list[str] = []
    used = 0
    for start, end in merge_windows(windows):
        prefix = "\n\n[... fragment pominięty ...]\n\n" if pieces else ""
        available = max_chars - used - len(prefix)
        if available <= 0:
            break
        fragment = text[start:end]
        if len(fragment) > available:
            fragment = fragment[:available]
        pieces.append(prefix + fragment)
        used += len(prefix) + len(fragment)
        if used >= max_chars:
            break

    return "".join(pieces), "fragmenty"


def build_prompt(
    record: dict[str, Any],
    assertions: list[dict[str, Any]],
    context: str,
    context_mode: str,
) -> str:
    name = str(record.get("nazwa", "") or "").strip()
    record_id = get_record_id(record)
    statistics = record.get("l_mk_statystyka")

    return f"""Zweryfikuj dane o liczbie ludnosci odczytane z hasla Slownika geograficznego Krolestwa Polskiego.

Korzystaj wylacznie z przekazanego tekstu. Nie uzywaj wiedzy wspolczesnej ani zewnetrznej. Sprawdz osobno kazda pozycje z listy TWIERDZENIA. Interesuje nas wylacznie liczba ludnosci miejscowosci lub jej czesci osadniczej, np. miasta, wsi, osady, kolonii albo folwarku. Nie interesuje nas laczna ludnosc parafii, dekanatu, gminy, powiatu ani innej jednostki zbiorczej. Wartosc "obecnie" oznacza czas opisywany przez autora hasla, a nie czasy wspolczesne.

SLOWNIK TYPOWYCH SKROTOW SGKP:
- "mk." i "mieszk." oznaczaja mieszkancow;
- "ludn.", "lud." i "ludu" oznaczaja ludnosc;
- "glow" w wyrazeniu dotyczacym ludnosci oznacza osoby, a nie gospodarstwa;
- "dusz", takze "dusz mez.", nie traktuj jako poszukiwanej liczby ludnosci;
- "dm." oznacza domy;
- samodzielny skrot "m." nie oznacza mieszkancow: oznacza morgi, czyli powierzchnie. Tak samo traktuj "mr." i "morg.";
- "kat.", "katol." oznaczaja katolikow, "ew.", "ewang." ewangelikow, a "izrael.", "zyd." ludnosc zydowska. W kontekscie ludnosci nie rozwijaj "kat." jako "katastralne" ani "ew." jako "ewidencyjne".

ZASADY INTERPRETACJI:
1. Nie dodawaj liczby wszystkich mieszkancow do jej podgrup. Przyklad: "7 mk., 5 ew., 2 kat." oznacza lacznie 7 mieszkancow, poniewaz 5 + 2 jest podzialem tej samej ludnosci.
2. Jezeli tekst nie podaje sumy, ale podaje kompletne rozlaczne skladniki w jednoznacznym kontekscie ludnosci miejscowosci, oblicz sume. Przyklad: "ludnosc: 270 kat., 155 ew." daje 425 mieszkancow. Uzyj wtedy statusu "zgodne_po_obliczeniu".
3. Analogicznie sumuj rozlaczne grupy wyznaniowe albo liczby mezczyzn i kobiet tylko wtedy, gdy kontekst wyraznie mowi o ludnosci tej samej miejscowosci. Nie sumuj danych z roznych miejscowosci, dat lub typow obiektow.
4. Liczby domow, morgow, dziesiecin, budynkow, osad, odleglosci, numerow stron i lat nie sa liczba ludnosci. Skrot "m." zawsze traktuj jako morgi, nigdy jako mieszkancow.
5. Nie uznawaj liczby "dusz" za liczbe ludnosci na potrzeby tego audytu, nawet gdy dotyczy miejscowosci. Przyklad: z tekstu "44 dusz mez. i 61 dz. ziemi" nie odczytuj zadnej poszukiwanej liczby ludnosci.
6. Jezeli twierdzenie JSON dotyczy parafii, dekanatu, gminy, powiatu, liczby dusz albo zostalo oparte na skrocie "m.", wybierz status "poza_zakresem". Nie zglaszaj takich danych w "pominiete_dane".
7. Gdy liczba ludnosci miejscowosci jest zgodna, ale tekst ogranicza jej znaczenie slowami "okolo", "blizko", "przeszlo", "do" albo "kilkadziesiat", uzyj statusu "zgodne_z_zastrzezeniem" i opisz zastrzezenie w uzasadnieniu.
8. Status "niezgodne" wybierz tylko wtedy, gdy dla miejscowosci z tekstu wynika inna liczba, inna data albo inna jednostka i potrafisz wskazac konkretny dowod. Sama obecnosc kwalifikatora nie jest niezgodnoscia.
9. Status "brak_w_tekscie" wybierz, gdy wartosci JSON nie da sie odnalezc ani jednoznacznie obliczyc. Jezeli OCR jest uszkodzony i nie pozwala na pewna ocene, wybierz "niejednoznaczne_ocr".
10. Dla wartosci null sprawdz, czy tekst rzeczywiscie podaje ludnosc miejscowosci za pomoca jednoznacznego okreslenia, np. "ludnosc", "ludn.", "mk." albo "mieszk.".
11. W "pominiete_dane" umieszczaj tylko odrebna informacje o ludnosci miejscowosci nieobecna w JSON. Nie zglaszaj danych parafii ani innych jednostek zbiorczych, liczby dusz, powierzchni oznaczonej "m." ani podgrup, jezeli JSON poprawnie zawiera ich laczna sume.

Dozwolone statusy weryfikacji:
- zgodne_doslowne
- zgodne_po_obliczeniu
- zgodne_z_zastrzezeniem
- zgodny_brak_danych
- niezgodne
- brak_w_tekscie
- niejednoznaczne_ocr
- poza_zakresem

Kazdy element weryfikacje musi zawierac indeks_statystyki i indeks_liczby zgodne z TWIERDZENIA. Zwroc dokladnie jeden element dla kazdego twierdzenia. Pole dowod ma byc krotkim, doslownym fragmentem tekstu (maksymalnie 300 znakow). Nie proponuj korekty, jesli nie wynika ona jasno z tekstu.

Nazwa: {name}
ID: {record_id}
Tryb kontekstu: {context_mode}

OBECNE l_mk_statystyka:
{json.dumps(statistics, ensure_ascii=False, indent=2)}

TWIERDZENIA:
{json.dumps(assertions, ensure_ascii=False, indent=2)}

TEKST HASLA:
{context}

Odpowiedz wylacznie poprawnym obiektem JSON:
{{
  "weryfikacje": [
    {{
      "indeks_statystyki": 1,
      "indeks_liczby": 1,
      "status": "zgodne_doslowne",
      "liczba_z_tekstu": "123",
      "sposob_weryfikacji": "wartosc podana wprost",
      "dowod": "123 mk.",
      "sugerowana_liczba": null,
      "uzasadnienie": "Krotkie uzasadnienie"
    }}
  ],
  "pominiete_dane": [
    {{
      "dotyczy": "nazwa obiektu",
      "data": "obecnie",
      "liczba": "123",
      "dowod": "krotki cytat",
      "uzasadnienie": "dlaczego jest to pominieta liczba ludnosci"
    }}
  ],
  "uwagi": ""
}}"""


def nullable_string(value: Any) -> str | None:
    if value is None:
        return None
    return str(value).strip()


def relaxed_literal(node: ast.AST) -> Any:
    """Odczytuje bez wykonywania kodu slownik w stylu Pythona/JSON."""

    if isinstance(node, ast.Expression):
        return relaxed_literal(node.body)
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.List):
        return [relaxed_literal(item) for item in node.elts]
    if isinstance(node, ast.Tuple):
        return [relaxed_literal(item) for item in node.elts]
    if isinstance(node, ast.Dict):
        return {
            relaxed_literal(key): relaxed_literal(value)
            for key, value in zip(node.keys, node.values, strict=True)
        }
    if isinstance(node, ast.Name) and node.id in {"null", "true", "false"}:
        return {"null": None, "true": True, "false": False}[node.id]
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, (ast.UAdd, ast.USub))
        and isinstance(node.operand, ast.Constant)
        and isinstance(node.operand.value, (int, float))
    ):
        value = node.operand.value
        return value if isinstance(node.op, ast.UAdd) else -value
    raise ValueError(f"Niedozwolona konstrukcja odpowiedzi: {type(node).__name__}")


def parse_population_model_json(output_text: str) -> dict[str, Any]:
    """Czyta JSON, a pomocniczo takze slownik z pojedynczymi apostrofami.

    Ollama Cloud nie wymusza obecnie structured outputs, dlatego model moze
    zwrocic skladnie podobna do literalu Pythona albo otoczyc JSON dodatkowym
    tekstem. Parser awaryjny obsluguje tylko dane (slowniki, listy, napisy,
    liczby i stale), nigdy nie wykonuje kodu.
    """

    stripped = output_text.strip()
    base_candidates = [stripped]
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end >= start:
        object_text = stripped[start : end + 1]
        if object_text != stripped:
            base_candidates.append(object_text)

    # Niektore odpowiedzi Gemmy zaczynaja sie od ``{{``, ale maja tylko
    # jedna klamre zamykajaca. To jednoznaczny blad formatowania: pierwsza
    # klamra nie moze rozpoczynac poprawnego obiektu JSON.
    candidates: list[str] = []
    for candidate in base_candidates:
        variants = [candidate]
        if candidate.startswith("{{"):
            variants.append(candidate[1:])
            if candidate.endswith("}}"):
                variants.append(candidate[1:-1])
        for variant in variants:
            if variant not in candidates:
                candidates.append(variant)

    errors: list[str] = []
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            errors.append(f"JSON: {exc}")
        else:
            if isinstance(parsed, dict):
                return parsed
            errors.append("JSON: element glowny nie jest obiektem")

        try:
            parsed = relaxed_literal(ast.parse(candidate, mode="eval"))
        except (SyntaxError, ValueError) as exc:
            errors.append(f"zapis tolerancyjny: {exc}")
        else:
            if isinstance(parsed, dict):
                return parsed
            errors.append("zapis tolerancyjny: element glowny nie jest obiektem")

    detail = errors[-1] if errors else "brak obiektu"
    raise ModelResponseError(
        f"Nie mozna odczytac odpowiedzi modelu ({detail})",
        raw_output=output_text,
    )


def normalize_model_result(
    parsed: dict[str, Any],
    assertions: list[dict[str, Any]],
) -> dict[str, Any]:
    raw_verifications = parsed.get("weryfikacje")
    if not isinstance(raw_verifications, list):
        raise ValueError("Odpowiedz nie zawiera listy weryfikacje")

    expected = {
        (item["indeks_statystyki"], item["indeks_liczby"]): item
        for item in assertions
    }
    received: dict[tuple[int, int], dict[str, Any]] = {}
    for raw in raw_verifications:
        if not isinstance(raw, dict):
            raise ValueError("Element weryfikacje nie jest obiektem")
        try:
            key = (int(raw["indeks_statystyki"]), int(raw["indeks_liczby"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Niepoprawne indeksy weryfikacji") from exc
        if key in received:
            raise ValueError(f"Powtorzona weryfikacja {key}")
        received[key] = raw

    if set(received) != set(expected):
        missing = sorted(set(expected) - set(received))
        extra = sorted(set(received) - set(expected))
        raise ValueError(f"Niezgodna lista weryfikacji; brak={missing}, nadmiar={extra}")

    verifications: list[dict[str, Any]] = []
    for key, assertion in expected.items():
        raw = received[key]
        status = str(raw.get("status", "") or "").strip()
        if status not in ALLOWED_ITEM_STATUSES:
            raise ValueError(f"Nieznany status modelu: {status!r}")
        verifications.append(
            {
                **assertion,
                "status": status,
                "liczba_z_tekstu": nullable_string(raw.get("liczba_z_tekstu")),
                "sposob_weryfikacji": nullable_string(raw.get("sposob_weryfikacji")),
                "dowod": nullable_string(raw.get("dowod")),
                "sugerowana_liczba": nullable_string(raw.get("sugerowana_liczba")),
                "uzasadnienie": nullable_string(raw.get("uzasadnienie")),
            }
        )

    raw_omissions = parsed.get("pominiete_dane", [])
    if not isinstance(raw_omissions, list):
        raise ValueError("Pole pominiete_dane nie jest lista")
    omissions: list[dict[str, Any]] = []
    for omission in raw_omissions:
        if not isinstance(omission, dict):
            raise ValueError("Element pominiete_dane nie jest obiektem")
        omissions.append(
            {
                "dotyczy": nullable_string(omission.get("dotyczy")),
                "data": nullable_string(omission.get("data")),
                "liczba": nullable_string(omission.get("liczba")),
                "dowod": nullable_string(omission.get("dowod")),
                "uzasadnienie": nullable_string(omission.get("uzasadnienie")),
            }
        )

    return {
        "weryfikacje": verifications,
        "pominiete_dane": omissions,
        "uwagi": nullable_string(parsed.get("uwagi")) or "",
    }


def record_status(result: dict[str, Any]) -> str:
    statuses = {item["status"] for item in result["weryfikacje"]}
    if "niezgodne" in statuses:
        return "niezgodny"
    if result["pominiete_dane"]:
        return "wymaga_sprawdzenia"
    if statuses & {
        "brak_w_tekscie",
        "niejednoznaczne_ocr",
        "zgodne_z_zastrzezeniem",
        "poza_zakresem",
    }:
        return "wymaga_sprawdzenia"
    if not statuses:
        return "zgodny_brak_danych"
    return "zgodny"


def verify_with_retries(
    ollama_url: str,
    api_key: str | None,
    model: str,
    prompt: str,
    assertions: list[dict[str, Any]],
    timeout: float,
    retries: int,
    retry_delay: float,
) -> dict[str, Any]:
    attempts = max(1, retries)
    for attempt in range(1, attempts + 1):
        output = ""
        try:
            output = ollama_chat(
                base_url=ollama_url,
                api_key=api_key,
                model=model,
                prompt=prompt,
                timeout=timeout,
            )
            parsed = parse_population_model_json(output)
            return normalize_model_result(parsed, assertions)
        except OllamaTransientError as exc:
            if attempt == attempts:
                raise
            print(
                f"Blad tymczasowy ({attempt}/{attempts}): {short_error(exc)}",
                file=sys.stderr,
            )
            time.sleep(max(0.0, retry_delay))
        except (json.JSONDecodeError, ValueError) as exc:
            if attempt == attempts:
                raw_output = getattr(exc, "raw_output", output)
                raise ModelResponseError(
                    f"Model nie zwrocil poprawnego schematu: {short_error(exc)}",
                    raw_output=raw_output,
                ) from exc
            print(
                f"Bledny JSON modelu ({attempt}/{attempts}): {short_error(exc)}",
                file=sys.stderr,
            )
            time.sleep(min(5.0, max(0.0, retry_delay)))

    raise RuntimeError("Nieudana weryfikacja modelu")


def compose_result_row(
    source_path: Path,
    record: dict[str, Any],
    model: str,
    validation: dict[str, Any],
    full_text: str,
    context: str,
    context_mode: str,
    result_source: str,
) -> dict[str, Any]:
    statistics = record.get("l_mk_statystyka")
    row = {
        "provider": "ollama",
        "plik_zrodlowy": source_path.name,
        "ID": get_record_id(record),
        "nazwa": record.get("nazwa"),
        "rodzaj": record.get("rodzaj"),
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "validation_key": validation_key(source_path.name, record, model),
        "text_sha256": sha256_text(full_text),
        "l_mk_sha256": sha256_text(canonical_json(statistics)),
        "dlugosc_tekstu": len(full_text),
        "dlugosc_kontekstu": len(context),
        "tryb_kontekstu": context_mode,
        "zrodlo_wyniku": result_source,
        "l_mk_statystyka": statistics,
        "czas_weryfikacji_utc": datetime.now(timezone.utc).isoformat(),
        **validation,
    }
    row["status_rekordu"] = record_status(row)
    return row


def make_result_row(
    source_path: Path,
    record: dict[str, Any],
    model: str,
    max_context_chars: int,
    snippet_radius: int,
    ollama_url: str,
    api_key: str | None,
    timeout: float,
    retries: int,
    retry_delay: float,
    sleep_seconds: float,
) -> dict[str, Any]:
    full_text = str(record.get("text", "") or "").strip()
    assertions = build_assertions(record)
    context, context_mode = prepare_text_context(
        full_text,
        assertions,
        max_chars=max_context_chars,
        radius=snippet_radius,
    )
    prompt = build_prompt(record, assertions, context, context_mode)
    validation = verify_with_retries(
        ollama_url=ollama_url,
        api_key=api_key,
        model=model,
        prompt=prompt,
        assertions=assertions,
        timeout=timeout,
        retries=retries,
        retry_delay=retry_delay,
    )
    if sleep_seconds > 0:
        time.sleep(sleep_seconds)

    return compose_result_row(
        source_path=source_path,
        record=record,
        model=model,
        validation=validation,
        full_text=full_text,
        context=context,
        context_mode=context_mode,
        result_source="ollama_api",
    )


def recover_result_row(
    source_path: Path,
    record: dict[str, Any],
    model: str,
    raw_output: str,
    max_context_chars: int,
    snippet_radius: int,
) -> dict[str, Any]:
    full_text = str(record.get("text", "") or "").strip()
    assertions = build_assertions(record)
    parsed = parse_population_model_json(raw_output)
    validation = normalize_model_result(parsed, assertions)
    context, context_mode = prepare_text_context(
        full_text,
        assertions,
        max_chars=max_context_chars,
        radius=snippet_radius,
    )
    return compose_result_row(
        source_path=source_path,
        record=record,
        model=model,
        validation=validation,
        full_text=full_text,
        context=context,
        context_mode=context_mode,
        result_source="odzyskany_z_logu_bledow",
    )


def output_paths(output_dir: Path, input_path: Path) -> dict[str, Path]:
    prefix = output_dir / f"{input_path.stem}_l_mk_weryfikacja"
    return {
        "partial": prefix.with_suffix(".partial.jsonl"),
        "final": prefix.with_suffix(".jsonl"),
        "csv": prefix.with_suffix(".csv"),
        "errors": prefix.with_suffix(".errors.jsonl"),
    }


def load_jsonl_by_key(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Niepoprawny JSONL w {path}:{line_number}") from exc
            key = str(row.get("validation_key", "") or "")
            if key:
                rows[key] = row
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False))
        handle.write("\n")


def error_row(
    input_path: Path,
    record: dict[str, Any],
    model: str,
    error: Exception,
) -> dict[str, Any]:
    row = {
        "plik_zrodlowy": input_path.name,
        "ID": get_record_id(record),
        "validation_key": validation_key(input_path.name, record, model),
        "blad": short_error(error),
        "czas_utc": datetime.now(timezone.utc).isoformat(),
    }
    raw_output = getattr(error, "raw_output", "")
    if raw_output:
        row["surowa_odpowiedz_modelu"] = raw_output
    return row


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


CSV_COLUMNS = [
    "plik_zrodlowy",
    "ID",
    "nazwa",
    "status_rekordu",
    "rodzaj_wiersza",
    "indeks_statystyki",
    "indeks_liczby",
    "dotyczy",
    "data",
    "liczba_json",
    "status",
    "liczba_z_tekstu",
    "sugerowana_liczba",
    "dowod",
    "uzasadnienie",
]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for result in rows:
            common = {
                "plik_zrodlowy": result.get("plik_zrodlowy"),
                "ID": result.get("ID"),
                "nazwa": result.get("nazwa"),
                "status_rekordu": result.get("status_rekordu"),
            }
            for item in result.get("weryfikacje", []):
                writer.writerow({**common, "rodzaj_wiersza": "weryfikacja", **item})
            for omission in result.get("pominiete_dane", []):
                writer.writerow(
                    {
                        **common,
                        "rodzaj_wiersza": "pominiete_dane",
                        "status": "pominiete_w_json",
                        **omission,
                    }
                )
    temporary.replace(path)


def process_file(
    input_path: Path,
    output_dir: Path,
    ollama_url: str,
    api_key: str | None,
    model: str,
    workers: int,
    limit: int | None,
    timeout: float,
    retries: int,
    retry_delay: float,
    sleep_seconds: float,
    max_context_chars: int,
    snippet_radius: int,
    include_empty: bool,
    skip_failed: bool,
    overwrite: bool,
) -> tuple[dict[str, Any], int]:
    started_at = time.monotonic()
    data = load_json(input_path)
    targets = [
        record
        for record in iter_target_records(data)
        if has_population_statistics(record, include_empty)
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = output_paths(output_dir, input_path)

    if overwrite:
        for path in paths.values():
            if path.exists():
                path.unlink()

    partial_rows = load_jsonl_by_key(paths["partial"])
    error_rows = load_jsonl_by_key(paths["errors"])
    current: list[tuple[dict[str, Any], str]] = []
    pending: list[dict[str, Any]] = []
    recovered = 0
    skipped_failed = 0
    for record in targets:
        record_id = get_record_id(record)
        if not record_id:
            raise RuntimeError(f"Rekord bez ID w {input_path}: {record.get('nazwa')}")
        key = validation_key(input_path.name, record, model)
        current.append((record, key))
        previous_error = error_rows.get(key)
        raw_output = (
            str(previous_error.get("surowa_odpowiedz_modelu", "") or "")
            if previous_error
            else ""
        )
        if key not in partial_rows and raw_output:
            try:
                recovered_row = recover_result_row(
                    source_path=input_path,
                    record=record,
                    model=model,
                    raw_output=raw_output,
                    max_context_chars=max_context_chars,
                    snippet_radius=snippet_radius,
                )
            except (SyntaxError, ValueError):
                pass
            else:
                append_jsonl(paths["partial"], recovered_row)
                partial_rows[key] = recovered_row
                error_rows.pop(key, None)
                recovered += 1
        if key not in partial_rows:
            if skip_failed and key in error_rows:
                skipped_failed += 1
            elif limit is None or len(pending) < limit:
                pending.append(record)

    print(
        f"{input_path.name}: rekordy={len(targets)}, nowe_zapytania={len(pending)}, "
        f"wznowione={sum(key in partial_rows for _, key in current)}, "
        f"odzyskane_z_logu={recovered}, pominiete_po_bledzie={skipped_failed}",
        file=sys.stderr,
    )

    completed = 0
    errors = 0
    worker_count = max(1, workers)
    if worker_count == 1:
        iterator = enumerate(pending, start=1)
        for index, record in iterator:
            print(
                f"[{input_path.name} {index}/{len(pending)}] {record_label(record)}",
                file=sys.stderr,
            )
            try:
                row = make_result_row(
                    input_path,
                    record,
                    model,
                    max_context_chars,
                    snippet_radius,
                    ollama_url,
                    api_key,
                    timeout,
                    retries,
                    retry_delay,
                    sleep_seconds,
                )
            except Exception as exc:
                errors += 1
                failed_row = error_row(input_path, record, model, exc)
                append_jsonl(paths["errors"], failed_row)
                error_rows[failed_row["validation_key"]] = failed_row
                print(
                    f"Blad {record_label(record)}: {short_error(exc)}",
                    file=sys.stderr,
                )
            else:
                append_jsonl(paths["partial"], row)
                partial_rows[row["validation_key"]] = row
                error_rows.pop(row["validation_key"], None)
                completed += 1
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(
                    make_result_row,
                    input_path,
                    record,
                    model,
                    max_context_chars,
                    snippet_radius,
                    ollama_url,
                    api_key,
                    timeout,
                    retries,
                    retry_delay,
                    sleep_seconds,
                ): record
                for record in pending
            }
            for future in as_completed(futures):
                record = futures[future]
                try:
                    row = future.result()
                except Exception as exc:
                    errors += 1
                    failed_row = error_row(input_path, record, model, exc)
                    append_jsonl(paths["errors"], failed_row)
                    error_rows[failed_row["validation_key"]] = failed_row
                    print(
                        f"[{input_path.name}] blad {record_label(record)}: "
                        f"{short_error(exc)}",
                        file=sys.stderr,
                    )
                else:
                    append_jsonl(paths["partial"], row)
                    partial_rows[row["validation_key"]] = row
                    error_rows.pop(row["validation_key"], None)
                    completed += 1
                    print(
                        f"[{input_path.name}] gotowe {completed}/{len(pending)} "
                        f"{record_label(record)}",
                        file=sys.stderr,
                    )

    final_rows = [partial_rows[key] for _, key in current if key in partial_rows]
    write_jsonl(paths["errors"], list(error_rows.values()))
    write_jsonl(paths["final"], final_rows)
    write_csv(paths["csv"], final_rows)
    statuses = Counter(row["status_rekordu"] for row in final_rows)
    summary = {
        "plik_zrodlowy": str(input_path),
        "rekordy_docelowe": len(targets),
        "nowe_zapytania": len(pending),
        "nowe_wyniki": completed,
        "odzyskane_z_logu_bledow": recovered,
        "pominiete_po_wczesniejszym_bledzie": skipped_failed,
        "bledy": errors,
        "nierozwiazane_bledy": sum(
            key in error_rows and key not in partial_rows for _, key in current
        ),
        "wyniki_lacznie": len(final_rows),
        "statusy": dict(statuses),
        "plik_jsonl": str(paths["final"]),
        "plik_csv": str(paths["csv"]),
        "czas": format_duration(time.monotonic() - started_at),
    }
    return summary, len(pending)


def run_dry_run(
    files: list[Path],
    include_empty: bool,
    limit: int | None,
    max_context_chars: int,
) -> int:
    remaining = limit
    total_targets = 0
    selected = 0
    long_texts = 0
    for path in files:
        data = load_json(path)
        targets = [
            record
            for record in iter_target_records(data)
            if has_population_statistics(record, include_empty)
        ]
        file_selected = len(targets) if remaining is None else min(len(targets), remaining)
        selected_records = targets[:file_selected]
        file_long = sum(
            max_context_chars > 0
            and len(str(record.get("text", "") or "")) > max_context_chars
            for record in selected_records
        )
        print(
            f"{path.name}: rekordy_docelowe={len(targets)}, "
            f"wybrane={file_selected}, dlugie_teksty={file_long}"
        )
        total_targets += len(targets)
        selected += file_selected
        long_texts += file_long
        if remaining is not None:
            remaining -= file_selected
            if remaining <= 0:
                break
    print(
        f"Razem: rekordy_docelowe={total_targets}, wybrane={selected}, "
        f"dlugie_teksty={long_texts}"
    )
    return 0


def main() -> int:
    started_at = time.monotonic()
    args = parse_args()
    files = input_files(args.input)
    missing = [path for path in files if not path.exists()]
    if missing:
        print(f"Nie znaleziono pliku: {missing[0]}", file=sys.stderr)
        return 2
    if not files:
        print(f"Nie znaleziono plikow JSON w {args.input}", file=sys.stderr)
        return 2

    if args.dry_run:
        return run_dry_run(
            files,
            include_empty=args.include_empty,
            limit=args.limit,
            max_context_chars=args.max_context_chars,
        )

    load_env_files(args.input)
    api_key = os.environ.get("OLLAMA_API_KEY")
    model = args.model or os.environ.get("OLLAMA_MODEL", DEFAULT_MODEL)
    ollama_url = args.ollama_url or os.environ.get(
        "OLLAMA_URL", DEFAULT_OLLAMA_URL
    )
    remaining = args.limit
    summaries: list[dict[str, Any]] = []
    total_errors = 0

    for path in files:
        if remaining is not None and remaining <= 0:
            break
        summary, requested = process_file(
            input_path=path,
            output_dir=args.output_dir,
            ollama_url=ollama_url,
            api_key=api_key,
            model=model,
            workers=args.workers,
            limit=remaining,
            timeout=args.timeout,
            retries=args.retries,
            retry_delay=args.retry_delay,
            sleep_seconds=args.sleep,
            max_context_chars=args.max_context_chars,
            snippet_radius=args.snippet_radius,
            include_empty=args.include_empty,
            skip_failed=args.skip_failed,
            overwrite=args.overwrite,
        )
        summaries.append(summary)
        total_errors += int(summary["bledy"])
        if remaining is not None:
            remaining -= requested

    args.output_dir.mkdir(parents=True, exist_ok=True)
    global_summary = {
        "provider": "ollama",
        "model": model,
        "ollama_url": ollama_url,
        "prompt_version": PROMPT_VERSION,
        "input": str(args.input),
        "output_dir": str(args.output_dir),
        "include_empty": args.include_empty,
        "skip_failed": args.skip_failed,
        "max_context_chars": args.max_context_chars,
        "pliki": summaries,
        "bledy_lacznie": total_errors,
        "czas_calkowity": format_duration(time.monotonic() - started_at),
    }
    summary_path = args.output_dir / "podsumowanie.json"
    write_json(summary_path, global_summary)

    for summary in summaries:
        print(summary["plik_jsonl"])
        print(summary["plik_csv"])
    print(summary_path)
    print(
        f"Czas wykonania calego przetwarzania: {global_summary['czas_calkowity']}",
        file=sys.stderr,
    )
    return 1 if total_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
