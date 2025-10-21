from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


logger = logging.getLogger(__name__)

PROTEIN_MATCH_TIERS = ("ACC_EXACT", "GENE_EXACT", "PNAME_UNIQUE", "AMBIGUOUS", "NO_MATCH")
SITE_MATCH_TIERS = ("SITE_EXACT", "SITE_NEAR±1", "NO_MATCH")

_GENE_SPLIT_PATTERN = re.compile(r"[;,]")
_NAME_PUNCT_PATTERN = re.compile(r"[^\w\s]")
_NAME_PARENS_PATTERN = re.compile(r"[()]+")
_NAME_ISOFORM_PATTERN = re.compile(r"isoform\s+\S+\s+of\s+", flags=re.IGNORECASE)

_REQUIRED_COLUMNS = {
    "protein_groups": {"protein_group_id", "uniprot_ids", "primary_uniprot", "gene_name", "protein_name", "source_file"},
    "hexnac_sites": {
        "site_id",
        "primary_uniprot",
        "position",
        "amino_acid",
        "sequence_window",
        "gene_name",
        "protein_name",
    },
    "phospho_sites": {"site_id", "primary_uniprot", "position", "amino_acid", "sequence_window"},
    "odb": {"uniprot_acc"},
}

# Column normalization for heterogeneous ODB CSV headers
_ODB_RENAME = {
    "oglcnac sites": "oglcnac_sites",
    "UniProtKB AC/ID": "uniprot_acc",
    "UniProtKB_AC/ID": "uniprot_acc",
    "Accession": "uniprot_acc",
    "Uniprot": "uniprot_acc",
    "UniProt": "uniprot_acc",
    "UniprotKB ID": "uniprot_acc",
    "Gene Symbol": "gene_symbol",
    "Gene": "gene_symbol",
    "GeneSymbol": "gene_symbol",
    "Protein names": "protein_name",
    "Protein Name": "protein_name",
    "ProteinNames": "protein_name",
    "Residue": "residue",
    "AA": "residue",
    "Amino Acid": "residue",
    "Position": "position",
    "Site": "position",
    "PMID": "pmid",
    "PMIDs": "pmid",
    "PMIDS": "pmid",
    "DOI": "doi",
    "Evidence": "evidence",
}

# Optional ODB columns we’ll create as empty if missing
_ODB_ALL_COLS = ["uniprot_acc","gene_symbol","protein_name","residue","position","pmid","doi","evidence","oglcnac_sites"]

MATCHABLE_PROTEIN_TIERS = {"ACC_EXACT", "GENE_EXACT", "PNAME_UNIQUE"}


def load_inputs(
    pg_path: str | Path,
    hex_path: str | Path,
    phos_path: str | Path,
    odb_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load pre-parsed inputs and validate required columns."""

    pg_df = _read_csv(pg_path, "protein_groups")
    hex_df = _read_csv(hex_path, "hexnac_sites")
    phos_df = _read_csv(phos_path, "phospho_sites")
    odb_df = _read_csv(odb_path, "odb")
    return pg_df, hex_df, phos_df, odb_df


def build_odb_indexes(odb_df: pd.DataFrame) -> dict[str, dict[Any, Any]]:
    """Prepare lookups that power deterministic best-effort matching."""

    _ensure_columns(odb_df, _REQUIRED_COLUMNS["odb"], "odb")

    by_acc: dict[str, dict[str, Any]] = {}
    by_gene: dict[str, list[dict[str, Any]]] = {}
    by_name_norm: dict[str, list[dict[str, Any]]] = {}
    by_site: dict[tuple[str, str, int], list[dict[str, Any]]] = {}

    for record in odb_df.to_dict(orient="records"):
        clean = _normalize_odb_record(record)
        acc = clean["uniprot_acc"]
        by_acc[acc] = clean

        gene_symbol = clean.get("gene_symbol")
        if gene_symbol:
            by_gene.setdefault(gene_symbol, []).append(clean)

        name_norm = _normalize_protein_name(clean.get("protein_name"))
        if name_norm:
            by_name_norm.setdefault(name_norm, []).append(clean)

        residue = clean.get("residue")
        position = clean.get("position")
        if residue and position is not None:
            by_site.setdefault((acc, residue, position), []).append(clean)

        # Parse compact site strings like "S132; T401" if present (fallback when explicit residue/position are absent)
        sites_str = clean.get("oglcnac_sites")
        if sites_str:
            for m in re.finditer(r"\b([ST])\s*([0-9]+)\b", str(sites_str)):
                aa = m.group(1)
                p = int(m.group(2))
                by_site.setdefault((acc, aa, p), []).append(clean)

    logger.info(
        "Built ODB indexes: %d proteins, %d genes, %d normalized names, %d site entries",
        len(by_acc),
        len(by_gene),
        len(by_name_norm),
        len(by_site),
    )
    return {"by_acc": by_acc, "by_gene": by_gene, "by_name_norm": by_name_norm, "by_site": by_site}


def match_proteins(pg_df: pd.DataFrame, idx: dict[str, dict[Any, Any]]) -> pd.DataFrame:
    """Assign the best-match ODB accession to each protein group."""

    _ensure_columns(pg_df, _REQUIRED_COLUMNS["protein_groups"], "protein_groups")
    result_rows: list[dict[str, Any]] = []
    by_acc = idx["by_acc"]
    by_gene = idx["by_gene"]
    by_name_norm = idx["by_name_norm"]

    for record in pg_df.to_dict(orient="records"):
        primary = str(record["primary_uniprot"]).upper().strip()
        gene_name = record.get("gene_name")
        protein_name = record.get("protein_name")
        row = {
            **record,
            "input_primary_uniprot": primary,
            "odb_uniprot": pd.NA,
            "match_level_protein": "NO_MATCH",
            "ambiguity_reason": pd.NA,
            "candidate_odb_uniprots": pd.NA,
            "no_match_reason": pd.NA,
        }

        acc_hit = by_acc.get(primary)
        if acc_hit:
            row["odb_uniprot"] = acc_hit["uniprot_acc"]
            row["match_level_protein"] = "ACC_EXACT"
            result_rows.append(row)
            continue

        gene_tokens = _extract_gene_tokens(gene_name)
        multi_gene_input = False
        if gene_tokens:
            if len(gene_tokens) > 1:
                multi_gene_input = True  # don't finalize yet; try name-based fallback below
            else:
                gene = gene_tokens[0]
                candidates = by_gene.get(gene, [])
                candidate_accs = sorted({cand["uniprot_acc"] for cand in candidates})
                if len(candidate_accs) == 1:
                    acc = candidate_accs[0]
                    row["odb_uniprot"] = acc
                    row["match_level_protein"] = "GENE_EXACT"
                    result_rows.append(row)
                    continue
                if candidate_accs:
                    row["match_level_protein"] = "AMBIGUOUS"
                    row["ambiguity_reason"] = "GENE_MULTI_ODB"
                    row["candidate_odb_uniprots"] = ",".join(candidate_accs)
                    result_rows.append(row)
                    continue
                row["no_match_reason"] = "GENE_NOT_IN_ODB"

        if pd.isna(protein_name) or str(protein_name).strip() == "":
            if pd.isna(row["no_match_reason"]):
                row["no_match_reason"] = "NO_NAME_AVAILABLE"
            result_rows.append(row)
            continue

        name_norm = _normalize_protein_name(protein_name)
        if not name_norm:
            if pd.isna(row["no_match_reason"]):
                row["no_match_reason"] = "NAME_NORMALIZED_EMPTY"
            result_rows.append(row)
            continue

        name_candidates = by_name_norm.get(name_norm, [])
        unique_accs = sorted({cand["uniprot_acc"] for cand in name_candidates})
        if len(unique_accs) == 1:
            acc = unique_accs[0]
            row["odb_uniprot"] = acc
            row["match_level_protein"] = "PNAME_UNIQUE"
            result_rows.append(row)
            continue

        if unique_accs:
            row["match_level_protein"] = "AMBIGUOUS"
            row["ambiguity_reason"] = "PNAME_MULTI_ODB"
            row["candidate_odb_uniprots"] = ",".join(unique_accs)
            result_rows.append(row)
            continue
        # No name hit; if multi-gene input, finalize as ambiguous; else as no-match
        if multi_gene_input:
            row["match_level_protein"] = "AMBIGUOUS"
            row["ambiguity_reason"] = "MULTI_GENE_INPUT"
            result_rows.append(row)
            continue
        if pd.isna(row["no_match_reason"]):
            row["no_match_reason"] = "NAME_NOT_IN_ODB"
        result_rows.append(row)

    columns = list(pg_df.columns) + [
        "input_primary_uniprot",
        "odb_uniprot",
        "match_level_protein",
        "ambiguity_reason",
        "candidate_odb_uniprots",
        "no_match_reason",
    ]
    result = pd.DataFrame(result_rows, columns=columns)
    return result


def match_hexnac_sites(
    hex_df: pd.DataFrame,
    protein_map: pd.DataFrame,
    idx: dict[str, dict[Any, Any]],
    site_slack: int = 1,
) -> pd.DataFrame:
    """Match HexNAc sites against ODB annotations with ±1 residue slack."""

    _ensure_columns(hex_df, _REQUIRED_COLUMNS["hexnac_sites"], "hexnac_sites")
    if "match_level_protein" not in protein_map.columns or "odb_uniprot" not in protein_map.columns:
        raise ValueError("protein_map must include match_level_protein and odb_uniprot columns")

    valid = protein_map[protein_map["match_level_protein"].isin(MATCHABLE_PROTEIN_TIERS)]
    lookup = {
        str(row["input_primary_uniprot"]): {
            "odb_uniprot": row["odb_uniprot"],
            "match_level": row["match_level_protein"],
        }
        for _, row in valid.iterrows()
        if pd.notna(row["odb_uniprot"])
    }

    results: list[dict[str, Any]] = []
    by_site = idx["by_site"]
    by_acc = idx["by_acc"]

    for record in hex_df.to_dict(orient="records"):
        primary = str(record["primary_uniprot"]).upper().strip()
        mapping = lookup.get(primary)
        if not mapping:
            continue

        odb_acc = mapping["odb_uniprot"]
        residue = _clean_residue(record.get("amino_acid"))
        position = _safe_int(record.get("position"))

        base_row = {
            "site_id": record["site_id"],
            "input_primary_uniprot": primary,
            "odb_uniprot": odb_acc,
            "amino_acid": residue if residue else pd.NA,
            "pos_input": position,
            "pos_odb": pd.NA,
            "site_match_tier": "NO_MATCH",
            "gene_name": pd.NA,
            "protein_name": pd.NA,
            "pmid": pd.NA,
            "doi": pd.NA,
            "evidence": pd.NA,
        }

        if residue and position is not None:
            exact_key = (odb_acc, residue, position)
            match_record = _first_or_none(by_site.get(exact_key, []))
            tier = "SITE_EXACT"
            if not match_record:
                match_record = _find_near_site(by_site, odb_acc, residue, position, site_slack)
                tier = "SITE_NEAR±1" if match_record else "NO_MATCH"

            if match_record:
                base_row["site_match_tier"] = tier
                base_row["pos_odb"] = match_record["position"]
                base_row["gene_name"] = match_record.get("gene_symbol") or pd.NA
                base_row["protein_name"] = match_record.get("protein_name") or pd.NA
                base_row["pmid"] = match_record.get("pmid") or pd.NA
                base_row["doi"] = match_record.get("doi") or pd.NA
                base_row["evidence"] = match_record.get("evidence") or pd.NA
            else:
                protein_record = by_acc.get(odb_acc)
                if protein_record:
                    base_row["gene_name"] = protein_record.get("gene_symbol") or pd.NA
                    base_row["protein_name"] = protein_record.get("protein_name") or pd.NA

        results.append(base_row)

    columns = [
        "site_id",
        "input_primary_uniprot",
        "odb_uniprot",
        "amino_acid",
        "pos_input",
        "pos_odb",
        "site_match_tier",
        "gene_name",
        "protein_name",
        "pmid",
        "doi",
        "evidence",
    ]
    return pd.DataFrame(results, columns=columns)


def overlap_phospho(phos_df: pd.DataFrame, protein_map: pd.DataFrame) -> pd.DataFrame:
    """Summarize phospho sites for proteins that matched the ODB."""

    _ensure_columns(phos_df, _REQUIRED_COLUMNS["phospho_sites"], "phospho_sites")
    if "match_level_protein" not in protein_map.columns or "odb_uniprot" not in protein_map.columns:
        raise ValueError("protein_map must include match_level_protein and odb_uniprot columns")

    valid = protein_map[protein_map["match_level_protein"].isin(MATCHABLE_PROTEIN_TIERS)]
    if valid.empty:
        return pd.DataFrame(
            columns=["input_primary_uniprot", "odb_uniprot", "match_level_protein", "n_phospho_sites"]
        )

    counts = (
        phos_df[phos_df["primary_uniprot"].isin(valid["input_primary_uniprot"])]
        .groupby("primary_uniprot")["site_id"]
        .nunique()
    )

    rows: list[dict[str, Any]] = []
    for _, prot in valid.iterrows():
        primary = prot["input_primary_uniprot"]
        if primary not in counts:
            continue
        rows.append(
            {
                "input_primary_uniprot": primary,
                "odb_uniprot": prot["odb_uniprot"],
                "match_level_protein": prot["match_level_protein"],
                "n_phospho_sites": int(counts[primary]),
            }
        )

    return pd.DataFrame(rows, columns=["input_primary_uniprot", "odb_uniprot", "match_level_protein", "n_phospho_sites"])


def emit_reports(
    proteins_overlap: pd.DataFrame,
    hex_matches: pd.DataFrame,
    phospho_overlap: pd.DataFrame,
    ambiguous: pd.DataFrame,
    no_match: pd.DataFrame,
    meta: dict[str, Any],
    out_dir: str | Path,
) -> None:
    """Persist report tables and metadata in the requested directory."""

    dir_path = Path(out_dir)
    dir_path.mkdir(parents=True, exist_ok=True)

    outputs = {
        "proteins_overlap.csv": proteins_overlap,
        "hexnac_site_matches.csv": hex_matches,
        "phospho_on_odb_proteins.csv": phospho_overlap,
        "unresolved_or_ambiguous.csv": ambiguous,
        "no_match.csv": no_match,
    }
    for name, frame in outputs.items():
        frame.to_csv(dir_path / name, index=False)

    meta_path = dir_path / "_meta_run.json"
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)


def main(
    pg_path: str | Path,
    hex_path: str | Path,
    phos_path: str | Path,
    odb_path: str | Path,
    out_dir: str | Path,
    site_slack: int = 1,
) -> int:
    """Entry point used by the CLI wrapper."""

    pg_df, hex_df, phos_df, odb_df = load_inputs(pg_path, hex_path, phos_path, odb_path)
    idx = build_odb_indexes(odb_df)
    protein_map = match_proteins(pg_df, idx)
    hex_matches = match_hexnac_sites(hex_df, protein_map, idx, site_slack=site_slack)
    phospho_overlap = overlap_phospho(phos_df, protein_map)

    proteins_overlap = protein_map[
        protein_map["match_level_protein"].isin(MATCHABLE_PROTEIN_TIERS) & protein_map["odb_uniprot"].notna()
    ][
        ["protein_group_id", "input_primary_uniprot", "odb_uniprot", "gene_name", "protein_name", "match_level_protein"]
    ].copy()

    ambiguous = protein_map[protein_map["match_level_protein"] == "AMBIGUOUS"][
        ["input_primary_uniprot", "gene_name", "protein_name", "ambiguity_reason", "candidate_odb_uniprots"]
    ].copy()

    no_match = protein_map[protein_map["match_level_protein"] == "NO_MATCH"][
        ["input_primary_uniprot", "gene_name", "protein_name", "no_match_reason"]
    ].copy()
    no_match.rename(columns={"no_match_reason": "reason"}, inplace=True)

    meta = {
        "inputs": {
            "protein_groups": str(Path(pg_path)),
            "hexnac_sites": str(Path(hex_path)),
            "phospho_sites": str(Path(phos_path)),
            "odb": str(Path(odb_path)),
        },
        "parameters": {"site_slack": site_slack},
        "counts": {
            "protein_matches": {tier: int((protein_map["match_level_protein"] == tier).sum()) for tier in PROTEIN_MATCH_TIERS},
            "site_matches": {
                "SITE_EXACT": int((hex_matches["site_match_tier"] == "SITE_EXACT").sum()),
                "SITE_NEAR±1": int((hex_matches["site_match_tier"] == "SITE_NEAR±1").sum()),
            },
        },
    }

    logger.info("Protein match counts: %s", meta["counts"]["protein_matches"])
    logger.info("Site match counts: %s", meta["counts"]["site_matches"])
    logger.info("Phospho overlap rows: %d", len(phospho_overlap))

    emit_reports(proteins_overlap, hex_matches, phospho_overlap, ambiguous, no_match, meta, out_dir)
    return 0


def _read_csv(path: str | Path, label: str) -> pd.DataFrame:
    frame = pd.read_csv(Path(path))
    if label == "odb":
        # Normalize headers
        if _ODB_RENAME:
            frame = frame.rename(columns={k: v for k, v in _ODB_RENAME.items() if k in frame.columns})
        # Ensure minimal required column exists
        if "uniprot_acc" not in frame.columns:
            raise ValueError(
                "odb CSV is missing a recognizable UniProt accession column. "
                "Tried aliases: 'UniProtKB AC/ID', 'Accession', 'Uniprot', 'UniProt'."
            )
        # Coerce types and add optional columns if missing
        if "position" in frame.columns:
            frame["position"] = pd.to_numeric(frame["position"], errors="coerce").astype("Int64")
        for c in _ODB_ALL_COLS:
            if c not in frame.columns:
                frame[c] = pd.NA
        # Final minimal check
        _ensure_columns(frame, _REQUIRED_COLUMNS["odb"], label)
        return frame
    else:
        _ensure_columns(frame, _REQUIRED_COLUMNS[label], label)
        return frame


def _ensure_columns(df: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"{label} is missing required columns: {sorted(missing)}")


def _normalize_odb_record(raw: dict[str, Any]) -> dict[str, Any]:
    def _clean(value: Any) -> Any:
        if pd.isna(value):
            return None
        if isinstance(value, str):
            text = value.strip()
            return text or None
        return value

    residue = _clean(raw.get("residue"))
    residue = residue.upper() if isinstance(residue, str) else None
    position = _safe_int(raw.get("position"))
    record = {
        "uniprot_acc": str(raw.get("uniprot_acc")).upper().strip(),
        "gene_symbol": (_clean(raw.get("gene_symbol")) or "").upper() or None,
        "protein_name": _clean(raw.get("protein_name")),
        "residue": residue,
        "position": position,
        "pmid": _clean(raw.get("pmid")),
        "doi": _clean(raw.get("doi")),
        "evidence": _clean(raw.get("evidence")),
        "oglcnac_sites": _clean(raw.get("oglcnac_sites")),
    }
    return record


def _normalize_protein_name(name: Any) -> str | None:
    if pd.isna(name):
        return None
    text = str(name).strip()
    if not text:
        return None
    text = _NAME_ISOFORM_PATTERN.sub("", text)
    text = _NAME_PARENS_PATTERN.sub(" ", text)
    text = _NAME_PUNCT_PATTERN.sub(" ", text.lower())
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def _extract_gene_tokens(gene_name: Any) -> list[str]:
    if pd.isna(gene_name):
        return []
    text = str(gene_name).strip()
    if not text:
        return []
    tokens = _GENE_SPLIT_PATTERN.split(text)
    unique: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        gene = token.strip().upper()
        if not gene:
            continue
        if gene not in seen:
            seen.add(gene)
            unique.append(gene)
    return unique


def _clean_residue(value: Any) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip().upper()
    if not text:
        return None
    return text


def _safe_int(value: Any) -> int | None:
    if isinstance(value, (int, float)) and not pd.isna(value):
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if text.isdigit():
            return int(text)
    return None


def _first_or_none(items: Iterable[dict[str, Any]] | None) -> dict[str, Any] | None:
    if not items:
        return None
    for item in items:
        return item
    return None


def _find_near_site(
    by_site: dict[tuple[str, str, int], list[dict[str, Any]]],
    odb_acc: str,
    residue: str,
    position: int,
    slack: int,
) -> dict[str, Any] | None:
    if slack <= 0:
        return None
    for offset in range(1, slack + 1):
        for delta in (-offset, offset):
            candidate = _first_or_none(by_site.get((odb_acc, residue, position + delta), []))
            if candidate:
                return candidate
    return None



__all__ = [
    "load_inputs",
    "build_odb_indexes",
    "match_proteins",
    "match_hexnac_sites",
    "overlap_phospho",
    "emit_reports",
    "main",
]


# Minimal CLI entrypoint
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Run O-GlcNAc matcher (human MVP).")
    p.add_argument("--protein_groups_path", default="data/processed/protein_groups_clean.csv")
    p.add_argument("--hexnac_sites_path", default="data/processed/hexnac_sites_clean.csv")
    p.add_argument("--phospho_sites_path", default="data/processed/phospho_sites_clean.csv")
    p.add_argument("--odb_path", default="data/odb/raw/homo-sapiens.csv")
    p.add_argument("--out_dir", default="data/processed/matches")
    p.add_argument("--site_slack", type=int, default=1)
    args = p.parse_args()
    raise SystemExit(main(
        pg_path=args.protein_groups_path,
        hex_path=args.hexnac_sites_path,
        phos_path=args.phospho_sites_path,
        odb_path=args.odb_path,
        out_dir=args.out_dir,
        site_slack=args.site_slack,
    ))
