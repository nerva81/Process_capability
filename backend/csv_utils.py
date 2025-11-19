# backend/csv_utils.py

from __future__ import annotations

import re
import io
from typing import Tuple

import pandas as pd


def detect_csv_format(raw_bytes: bytes) -> Tuple[str, str]:
    """
    Hrubá detekce oddělovače a desetinného znaku z prvních pár řádků.

    Vrací:
        (delimiter, decimal)
        delimiter: např. ';', ',', '\\t', ' ' (whitespace)
        decimal: '.' nebo ','
    """
    text = raw_bytes.decode("utf-8", errors="ignore")
    lines = text.splitlines()
    sample = "\n".join(lines[:10])

    # Kandidáti na oddělovač – preferujeme ;, pak tab, pak čárku, jinak whitespace
    candidates = [";", "\t", ","]
    counts = {c: sample.count(c) for c in candidates}

    if counts[";"] > 0:
        delimiter = ";"
    elif counts["\t"] > 0:
        delimiter = "\t"
    elif counts[","] > 0:
        delimiter = ","
    else:
        delimiter = " "  # whitespace

    # Detekce desetinné tečky/čárky
    decimal = "."
    if delimiter != ",":
        if re.search(r"\d,\d", sample):
            decimal = ","
        elif re.search(r"\d\.\d", sample):
            decimal = "."

    # Typická EU kombinace ; + ,
    if delimiter == ";" and decimal == ".":
        decimal = ","

    return delimiter, decimal


def force_numeric(series: pd.Series) -> pd.Series:
    """
    Robustní konverze na číslo – ošetří i čárky jako desetinný oddělovač.
    Nepadá na chybě, chybná hodnota → NaN.
    """
    s = (
        series.astype(str)
        .str.strip()
        .str.replace(",", ".", regex=False)
    )
    return pd.to_numeric(s, errors="coerce")
