from __future__ import annotations

import logging
import re
from typing import Any, Callable

import pandas as pd


logger = logging.getLogger(__name__)


def split_uniprot_ids(value: Any) -> list[str]:
    """Split a UniProt ID field into canonical identifiers."""
    if pd.isna(value) or value == "":
        return []
    return [_strip_isoform(part.strip()) for part in str(value).split(";") if part.strip()]


def parse_positions(value: Any) -> list[int]:
    """Parse residue positions from a delimited string."""
    if pd.isna(value) or value == "":
        return []
    positions: list[int] = []
    for token in re.split(r"[;,]", str(value)):
        token = token.strip()
        if token.isdigit():
            positions.append(int(token))
        elif token:
            logger.warning("Skipping malformed position token '%s' in value '%s'", token, value)
    return positions


def clean_sequence_window(value: Any) -> str | None:
    """Normalize sequence window strings."""
    if pd.isna(value):
        return None
    text = str(value).strip().upper()
    return text or None


def normalize_columns(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    """Rename DataFrame columns according to *mapping*."""
    return df.rename(columns=mapping, errors="ignore")


def parse_protein_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize protein group annotations for downstream matching."""
    mapping = {"Protein IDs": "uniprot_ids", "Gene names": "gene_name", "Protein names": "protein_name"}

    before = len(df)
    frame = normalize_columns(df.copy(), mapping)
    _map_or_default(frame, "uniprot_ids", split_uniprot_ids, list)

    frame = _finalize_frame(frame)
    after = len(frame)
    logger.info("Parsed protein_groups: %d rows -> %d rows", before, after)
    return frame


def parse_hexnac_sites(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize HexNAc site annotations for downstream matching."""
    mapping = {
        "Proteins": "uniprot_ids",
        "Positions within proteins": "positions",
        "Amino acid": "amino_acid",
        "Sequence window": "sequence_window",
        "Leading proteins": "protein_name",
        "Gene names": "gene_name",
    }

    before = len(df)
    frame = normalize_columns(df.copy(), mapping)
    _map_or_default(frame, "uniprot_ids", split_uniprot_ids, list)
    _map_or_default(frame, "positions", parse_positions, list)
    _map_or_default(frame, "amino_acid", _normalize_amino_acid, lambda: None)
    _map_or_default(frame, "sequence_window", clean_sequence_window, lambda: None)
    frame = _finalize_frame(frame)
    after = len(frame)
    logger.info("Parsed hexnac_sites: %d rows -> %d rows", before, after)
    return frame


def parse_phospho_sites(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize phospho site annotations for downstream comparison."""
    mapping = {
        "Proteins": "uniprot_ids",
        "Positions within proteins": "positions",
        "Amino acid": "amino_acid",
        "Sequence window": "sequence_window",
    }

    before = len(df)
    frame = normalize_columns(df.copy(), mapping)
    _map_or_default(frame, "uniprot_ids", split_uniprot_ids, list)
    _map_or_default(frame, "positions", parse_positions, list)
    _map_or_default(frame, "amino_acid", _normalize_amino_acid, lambda: None)
    _map_or_default(frame, "sequence_window", clean_sequence_window, lambda: None)
    frame = _finalize_frame(frame)
    after = len(frame)
    logger.info("Parsed phospho_sites: %d rows -> %d rows", before, after)
    return frame


def _strip_isoform(uniprot_id: str) -> str:
    return uniprot_id.split("-")[0]


def _normalize_amino_acid(value: Any) -> str | None:
    if pd.isna(value):
        return None
    amino = str(value).strip().upper()
    if amino and amino not in {"S", "T", "Y"}:
        logger.warning("Encountered non-standard amino acid '%s'", amino)
    return amino or None


def _finalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    canonical = ("uniprot_ids", "gene_name", "protein_name", "positions", "amino_acid", "sequence_window")
    for column in canonical:
        if column not in frame.columns:
            frame[column] = None

    frame["uniprot_ids"] = frame["uniprot_ids"].map(lambda v: v if isinstance(v, list) else [])
    frame["positions"] = frame["positions"].map(lambda v: v if isinstance(v, list) else [])

    deduped = frame.drop_duplicates()
    dropped = len(frame) - len(deduped)
    if dropped:
        logger.info("Dropped %d duplicate rows", dropped)
    return deduped


def _map_or_default(
    frame: pd.DataFrame,
    column: str,
    func: Callable[[Any], Any],
    default_factory: Callable[[], Any],
) -> None:
    if column in frame:
        frame[column] = frame[column].map(func)
    else:
        frame[column] = [default_factory() for _ in range(len(frame))]


__all__ = ["parse_protein_groups", "parse_hexnac_sites", "parse_phospho_sites"]
