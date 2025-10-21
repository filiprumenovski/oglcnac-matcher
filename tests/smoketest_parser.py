"""
Lightweight smoketest for Fehl Lab O-GlcNAc parser pipeline.
Run from repo root: `pytest -q tests/smoketest_parser.py`  or `python tests/smoketest_parser.py`
"""

import os
import pandas as pd

# --- expected output paths ---
OUT_DIR = "data/processed"
FILES = [
    "protein_groups_clean.csv",
    "hexnac_sites_clean.csv",
    "phospho_sites_clean.csv",
]

REQUIRED_COLS = {
    "protein_groups_clean.csv": ["primary_uniprot", "uniprot_ids", "gene_name", "protein_name", "source_file"],
    "hexnac_sites_clean.csv": ["site_id", "primary_uniprot", "position", "amino_acid", "source_file"],
    "phospho_sites_clean.csv": ["site_id", "primary_uniprot", "position", "amino_acid", "source_file"],
}

def read_csv_safe(path):
    try:
        return pd.read_csv(path)
    except Exception as e:
        raise AssertionError(f"❌ Failed to read {path}: {e}")

def test_outputs_exist_and_loadable():
    for f in FILES:
        p = os.path.join(OUT_DIR, f)
        assert os.path.exists(p), f"❌ Missing output file: {p}"
        df = read_csv_safe(p)
        assert len(df) > 0, f"❌ {f} is empty"
        for col in REQUIRED_COLS[f]:
            assert col in df.columns, f"❌ {f} missing column {col}"
    print("✅ All core output files exist, load, and have required columns.")

def test_basic_schema_types():
    hx = read_csv_safe(os.path.join(OUT_DIR, "hexnac_sites_clean.csv"))
    ph = read_csv_safe(os.path.join(OUT_DIR, "phospho_sites_clean.csv"))

    for name, df in {"HexNAc": hx, "Phospho": ph}.items():
        # positions must be numeric
        assert pd.api.types.is_numeric_dtype(df["position"]), f"❌ {name} positions not numeric"
        # amino acids should be 1-char
        bad = df["amino_acid"].dropna().map(lambda x: len(str(x)) != 1).sum()
        assert bad == 0, f"❌ {name} amino_acid contains multi-char entries"
    print("✅ Position + amino_acid columns valid in both site tables.")

def test_unmapped_report_if_enabled():
    path = os.path.join(OUT_DIR, "unmapped_report.csv")
    if os.path.exists(path):
        df = read_csv_safe(path)
        assert {"table","reason"}.issubset(df.columns), "❌ unmapped_report missing required cols"
        print(f"ℹ️  Unmapped report exists with {len(df)} rows.")
    else:
        print("ℹ️  No unmapped report found (ok if no rows were dropped).")

if __name__ == "__main__":
    test_outputs_exist_and_loadable()
    test_basic_schema_types()
    test_unmapped_report_if_enabled()
    print("✅ Smoketest finished clean.")