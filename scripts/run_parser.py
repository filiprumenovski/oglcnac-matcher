#!/usr/bin/env python3
"""
Simple runner for Fehl Lab O-GlcNAc parser.
Usage:
    python scripts/run_parser.py
"""
import os
import sys
from pathlib import Path
import pandas as pd

# --- Make sure "src" is on the import path ---
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
# ---------------------------------------------

from oglcnac_matcher.parsers import (
    parse_protein_groups,
    parse_hexnac_sites,
    parse_phospho_sites,
)

def to_df(result):
    """Extract the real pandas DataFrame from a ParseResult."""
    if isinstance(result, pd.DataFrame):
        return result
    for attr in ("records", "df", "frame", "data", "dataframe", "table"):
        if hasattr(result, attr):
            val = getattr(result, attr)
            if isinstance(val, pd.DataFrame):
                return val
    raise TypeError(f"Cannot extract DataFrame from {type(result)}")

def main():
    print("🚀 Running Fehl Lab parser on data/raw/")

    pg = pd.read_excel("data/raw/protein-groups.xlsx")
    hx = pd.read_excel("data/raw/HexNAc-sites.xlsx")
    ph = pd.read_excel("data/raw/phospho-sites.xlsx")

    pg_df = to_df(parse_protein_groups(pg))
    hx_df = to_df(parse_hexnac_sites(hx))
    ph_df = to_df(parse_phospho_sites(ph))

    out = ROOT / "data" / "processed"
    out.mkdir(parents=True, exist_ok=True)

    pg_df.to_csv(out / "protein_groups_clean.csv", index=False)
    hx_df.to_csv(out / "hexnac_sites_clean.csv", index=False)
    ph_df.to_csv(out / "phospho_sites_clean.csv", index=False)

    print("\n✅ Parser complete — cleaned CSVs written to data/processed/")
    for name in ("protein_groups_clean.csv",
                 "hexnac_sites_clean.csv",
                 "phospho_sites_clean.csv"):
        print("   •", name)

if __name__ == "__main__":
    main()
