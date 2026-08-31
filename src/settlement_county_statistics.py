#!/usr/bin/env python3
"""Zlicza miejscowosci i ich pokrycie powiatami w plikach JSON SGKP.

Skrypt analizuje rekordy rodzaju ``indywidualne`` oraz elementy rekordow
rodzaju ``zbiorcze``. Rodzic hasla zbiorczego nie jest miejscowoscia i nie
jest liczony. Za miejscowosc uznawany jest analizowany obiekt zawierajacy
klucz ``typ_punktu_osadniczego``.

Miejscowosc nalezy do zakresu projektu, jezeli ma niepusta wartosc pola
``powiat_ujednolicony`` i nie zawiera klucza ``powiat_uwagi``.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterator


DEFAULT_PATTERN = "sgkp_*.json"
CSV_NAME = "settlement_county_statistics.csv"
JSON_NAME = "settlement_county_statistics.json"


@dataclass
class Statistics:
    """Liczniki dla pojedynczego pliku albo calego zbioru."""

    plik: str
    analizowane_rekordy_indywidualne: int = 0
    analizowane_elementy_zbiorcze: int = 0
    miejscowosci_lacznie: int = 0
    miejscowosci_rekordy_indywidualne: int = 0
    miejscowosci_elementy_zbiorcze: int = 0
    miejscowosci_w_zakresie_projektu: int = 0
    miejscowosci_z_powiat_ujednolicony: int = 0
    miejscowosci_z_powiat_ujednolicony_i_powiat_uwagi: int = 0
    miejscowosci_bez_powiat_ujednolicony: int = 0
    miejscowosci_z_powiat_uwagi_bez_powiat_ujednolicony: int = 0
    puste_typ_punktu_osadniczego: int = 0
    puste_powiat_ujednolicony: int = 0
    puste_powiat_uwagi: int = 0

    @property
    def odsetek_w_zakresie_projektu(self) -> float:
        if not self.miejscowosci_lacznie:
            return 0.0
        return round(
            100 * self.miejscowosci_w_zakresie_projektu
            / self.miejscowosci_lacznie,
            2,
        )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["odsetek_w_zakresie_projektu"] = self.odsetek_w_zakresie_projektu
        return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Zlicza miejscowosci (rekordy indywidualne i elementy hasel "
            "zbiorczych z polem typ_punktu_osadniczego) oraz miejscowosci "
            "majace powiat_ujednolicony bez pola powiat_uwagi."
        )
    )
    parser.add_argument(
        "input_dir",
        nargs="?",
        type=Path,
        default=Path("sgkp_przekazane"),
        help="Katalog z plikami JSON (domyslnie: sgkp_przekazane).",
    )
    parser.add_argument(
        "--pattern",
        default=DEFAULT_PATTERN,
        help=f"Wzorzec nazw plikow (domyslnie: {DEFAULT_PATTERN}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("sgkp_statystyki"),
        help="Katalog raportow CSV i JSON (domyslnie: sgkp_statystyki).",
    )
    return parser.parse_args()


def natural_sort_key(path: Path) -> tuple[Any, ...]:
    """Sortuje sgkp_2 przed sgkp_10, rowniez dla innych nazw."""

    return tuple(
        int(part) if part.isdigit() else part.casefold()
        for part in re.split(r"(\d+)", path.name)
    )


def has_nonempty_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, dict, tuple, set)):
        return bool(value)
    return True


def load_json(path: Path) -> list[Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("glowna struktura JSON nie jest lista")
    return data


def iter_objects(data: list[Any]) -> Iterator[tuple[str, dict[str, Any]]]:
    """Zwraca obiekty podlegajace analizie wraz z rodzajem ich pochodzenia."""

    for record in data:
        if not isinstance(record, dict):
            continue

        kind = record.get("rodzaj")
        if kind == "indywidualne":
            yield "indywidualne", record
            continue

        if kind != "zbiorcze":
            continue

        elements = record.get("elementy")
        if not isinstance(elements, list):
            continue
        for element in elements:
            if isinstance(element, dict):
                yield "element_zbiorczy", element


def update_statistics(
    stats: Statistics,
    source_kind: str,
    record: dict[str, Any],
) -> None:
    if source_kind == "indywidualne":
        stats.analizowane_rekordy_indywidualne += 1
    else:
        stats.analizowane_elementy_zbiorcze += 1

    # Obecnosc klucza jest kryterium podanym dla miejscowosci. Puste wartosci
    # sa dodatkowo raportowane, aby mozna bylo wykryc problem jakosci danych.
    if "typ_punktu_osadniczego" not in record:
        return

    stats.miejscowosci_lacznie += 1
    if source_kind == "indywidualne":
        stats.miejscowosci_rekordy_indywidualne += 1
    else:
        stats.miejscowosci_elementy_zbiorcze += 1

    if not has_nonempty_value(record.get("typ_punktu_osadniczego")):
        stats.puste_typ_punktu_osadniczego += 1

    has_unified_key = "powiat_ujednolicony" in record
    has_unified = has_nonempty_value(record.get("powiat_ujednolicony"))
    has_notes_key = "powiat_uwagi" in record
    has_notes = has_nonempty_value(record.get("powiat_uwagi"))

    if has_unified:
        stats.miejscowosci_z_powiat_ujednolicony += 1
        if has_notes_key:
            stats.miejscowosci_z_powiat_ujednolicony_i_powiat_uwagi += 1
        else:
            stats.miejscowosci_w_zakresie_projektu += 1
    else:
        stats.miejscowosci_bez_powiat_ujednolicony += 1

    if has_notes and not has_unified:
        stats.miejscowosci_z_powiat_uwagi_bez_powiat_ujednolicony += 1
    if has_unified_key and not has_unified:
        stats.puste_powiat_ujednolicony += 1
    if has_notes_key and not has_notes:
        stats.puste_powiat_uwagi += 1


def analyze_file(path: Path) -> Statistics:
    stats = Statistics(plik=path.name)
    for source_kind, record in iter_objects(load_json(path)):
        update_statistics(stats, source_kind, record)
    return stats


def sum_statistics(rows: list[Statistics]) -> Statistics:
    total = Statistics(plik="RAZEM")
    for row in rows:
        for field_name in Statistics.__dataclass_fields__:
            if field_name != "plik":
                setattr(
                    total,
                    field_name,
                    getattr(total, field_name) + getattr(row, field_name),
                )
    return total


def write_csv_atomic(path: Path, rows: list[Statistics], total: Statistics) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fieldnames = list(total.to_dict())
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in [*rows, total]:
                writer.writerow(row.to_dict())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_json_atomic(
    path: Path,
    input_dir: Path,
    pattern: str,
    rows: list[Statistics],
    total: Statistics,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    report = {
        "czas_utc": datetime.now(timezone.utc).isoformat(),
        "katalog_zrodlowy": str(input_dir.resolve()),
        "wzorzec_plikow": pattern,
        "definicje": {
            "miejscowosc": (
                "rekord indywidualny lub element rekordu zbiorczego "
                "z kluczem typ_punktu_osadniczego"
            ),
            "w_zakresie_projektu": (
                "niepuste powiat_ujednolicony i brak klucza powiat_uwagi"
            ),
        },
        "pliki": [row.to_dict() for row in rows],
        "razem": total.to_dict(),
    }
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def print_table(rows: list[Statistics], total: Statistics) -> None:
    shown = [*rows, total]
    name_width = max(len("Plik"), *(len(row.plik) for row in shown))
    headings = (
        f"{'Plik':<{name_width}}  {'Miejscowosci':>12}  {'Indywidualne':>13}  "
        f"{'Elementy':>9}  {'W projekcie':>12}  {'Udzial':>7}"
    )
    print(headings)
    print("-" * len(headings))
    for row in shown:
        print(
            f"{row.plik:<{name_width}}  "
            f"{row.miejscowosci_lacznie:>12}  "
            f"{row.miejscowosci_rekordy_indywidualne:>13}  "
            f"{row.miejscowosci_elementy_zbiorcze:>9}  "
            f"{row.miejscowosci_w_zakresie_projektu:>12}  "
            f"{row.odsetek_w_zakresie_projektu:>6.2f}%"
        )


def main() -> int:
    args = parse_args()

    if not args.input_dir.is_dir():
        print(f"Nie znaleziono katalogu: {args.input_dir}", file=sys.stderr)
        return 2

    paths = sorted(args.input_dir.glob(args.pattern), key=natural_sort_key)
    paths = [path for path in paths if path.is_file()]
    if not paths:
        print(
            f"Brak plikow pasujacych do {args.pattern!r} w {args.input_dir}",
            file=sys.stderr,
        )
        return 2

    rows: list[Statistics] = []
    try:
        for path in paths:
            rows.append(analyze_file(path))
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"Nie mozna przeanalizowac {path}: {exc}", file=sys.stderr)
        return 2

    total = sum_statistics(rows)
    csv_path = args.output_dir / CSV_NAME
    json_path = args.output_dir / JSON_NAME
    try:
        write_csv_atomic(csv_path, rows, total)
        write_json_atomic(
            json_path,
            args.input_dir,
            args.pattern,
            rows,
            total,
        )
    except OSError as exc:
        print(f"Nie mozna zapisac raportu: {exc}", file=sys.stderr)
        return 2

    print_table(rows, total)
    print()
    print(f"CSV:  {csv_path}")
    print(f"JSON: {json_path}")

    quality_warnings = (
        total.puste_typ_punktu_osadniczego
        + total.puste_powiat_ujednolicony
        + total.puste_powiat_uwagi
    )
    if quality_warnings:
        print(
            "Uwaga: wykryto puste pola: "
            f"typ_punktu_osadniczego={total.puste_typ_punktu_osadniczego}, "
            f"powiat_ujednolicony={total.puste_powiat_ujednolicony}, "
            f"powiat_uwagi={total.puste_powiat_uwagi}.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
