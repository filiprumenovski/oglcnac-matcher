from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import pandas as pd


logger = logging.getLogger(__name__)

_ID_SPLIT_PATTERN = re.compile(r"[;,]")
_UNMAPPED_REASONS = {
    "EMPTY_ID",
    "MALFORMED_ID",
    "NO_MATCH_IN_CACHE",
    "SPECIES_MISMATCH",
    "NO_POSITION_FOR_SITE",
}
_HEX_ALLOWED = {"S", "T"}
_PHOS_ALLOWED = {"S", "T", "Y"}


@dataclass(frozen=True)
class ParseResult:
    """Container for parsed classifier tables."""

    records: pd.DataFrame
    unmapped: pd.DataFrame


def split_uniprot_ids(value: Any) -> list[str]:
    """Normalize UniProt identifiers according to the global contract."""

    tokens: Iterable[Any]
    if isinstance(value, list):
        tokens = value
    elif pd.isna(value) or value == "":
        tokens = []
    else:
        tokens = _ID_SPLIT_PATTERN.split(str(value))

    seen: set[str] = set()
    result: list[str] = []
    for token in tokens:
        text = str(token).strip()
        if not text:
            continue
        canonical = text.upper()
        if canonical not in seen:
            seen.add(canonical)
            result.append(canonical)
    return result


def parse_positions(value: Any) -> list[int]:
    """Extract integer residue positions from a delimited field."""

    tokens: Iterable[Any]
    if isinstance(value, list):
        tokens = value
    elif pd.isna(value) or value == "":
        tokens = []
    else:
        tokens = _ID_SPLIT_PATTERN.split(str(value))

    positions: list[int] = []
    for token in tokens:
        text = str(token).strip()
        if not text:
            continue
        if text.isdigit():
            positions.append(int(text))
        else:
            logger.warning("Skipping malformed position token '%s' in value '%s'", text, value)
    return positions


def clean_sequence_window(value: Any) -> str | None:
    """Uppercase sequence windows and treat empties as missing."""

    if pd.isna(value) or value == "":
        return None
    text = str(value).strip().upper()
    return text or None


def parse_protein_groups(
    df: pd.DataFrame,
    source_file: str = "protein-groups.xlsx",
) -> ParseResult:
    """Produce protein-level records ready for classifier joins."""

    aliases = {
        "uniprot_ids": ("Protein IDs", "Majority protein IDs"),
        "gene_name": ("Gene names",),
        "protein_name": ("Protein names",),
    }
    frame = _rename_with_candidates(df.copy(), aliases, required=("uniprot_ids",))

    before = len(frame)
    frame["uniprot_ids"] = frame["uniprot_ids"].apply(split_uniprot_ids)

    unmapped_rows: list[dict[str, Any]] = []
    frame = _drop_empty_ids(frame, source_file, "protein_groups", unmapped_rows)

    frame["primary_uniprot"] = frame["uniprot_ids"].apply(_derive_primary_uniprot)
    frame, malformed = _drop_null_primary(frame, source_file, "protein_groups")
    unmapped_rows.extend(malformed)

    _ensure_columns(frame, ("gene_name", "protein_name"))
    frame["source_file"] = source_file

    ordered = [
        "uniprot_ids",
        "primary_uniprot",
        "gene_name",
        "protein_name",
        "source_file",
    ]
    ordered += [col for col in frame.columns if col not in ordered]

    frame = frame[ordered].drop_duplicates(subset=["primary_uniprot"]).reset_index(drop=True)
    frame.insert(0, "protein_group_id", [f"PG{i:06d}" for i in range(1, len(frame) + 1)])
    logger.info("Parsed protein_groups: %d rows -> %d rows", before, len(frame))

    report = _build_unmapped_report(unmapped_rows)
    return ParseResult(records=frame, unmapped=report)


def parse_hexnac_sites(
    df: pd.DataFrame,
    source_file: str = "HexNAc-sites.xlsx",
) -> ParseResult:
    """Produce HexNAc site records expanded to single-position rows."""

    aliases = {
        "uniprot_ids": ("Leading proteins", "Proteins"),
        "positions": ("Positions within proteins",),
        "amino_acid": ("Amino acid",),
        "sequence_window": ("Sequence window",),
        "gene_name": ("Gene names",),
        "protein_name": ("Protein names",),
    }
    frame = _rename_with_candidates(df.copy(), aliases, required=("uniprot_ids", "positions"))

    before = len(frame)
    frame["uniprot_ids"] = frame["uniprot_ids"].apply(split_uniprot_ids)
    unmapped_rows: list[dict[str, Any]] = []
    frame = _drop_empty_ids(frame, source_file, "hexnac_sites", unmapped_rows)

    frame["positions"] = frame["positions"].apply(parse_positions)
    frame = _drop_empty_positions(frame, source_file, "hexnac_sites", unmapped_rows)

    frame["amino_acid"] = frame.get("amino_acid", pd.Series(index=frame.index)).apply(
        _normalize_amino_acid(_HEX_ALLOWED)
    )
    frame["sequence_window"] = frame.get("sequence_window", pd.Series(index=frame.index)).apply(clean_sequence_window)

    _ensure_columns(frame, ("gene_name", "protein_name"))

    frame = frame.explode("positions", ignore_index=True)
    frame.rename(columns={"positions": "position"}, inplace=True)
    frame["position"] = frame["position"].astype(int)

    frame["primary_uniprot"] = frame["uniprot_ids"].apply(_derive_primary_uniprot)
    frame, malformed = _drop_null_primary(frame, source_file, "hexnac_sites")
    unmapped_rows.extend(malformed)

    frame["source_file"] = source_file

    ordered = [
        "uniprot_ids",
        "primary_uniprot",
        "position",
        "amino_acid",
        "sequence_window",
        "gene_name",
        "protein_name",
        "source_file",
    ]
    ordered += [col for col in frame.columns if col not in ordered]

    frame = frame[ordered].drop_duplicates(subset=["primary_uniprot", "position", "amino_acid"]).reset_index(drop=True)
    frame.insert(0, "site_id", [f"HX{i:06d}" for i in range(1, len(frame) + 1)])
    logger.info("Parsed hexnac_sites: %d rows -> %d rows", before, len(frame))

    report = _build_unmapped_report(unmapped_rows)
    return ParseResult(records=frame, unmapped=report)


def parse_phospho_sites(
    df: pd.DataFrame,
    source_file: str = "phospho-sites.xlsx",
) -> ParseResult:
    """Produce phospho site records aligned with classifier expectations."""

    aliases = {
        "uniprot_ids": ("Leading proteins", "Proteins"),
        "positions": ("Positions within proteins",),
        "amino_acid": ("Amino acid",),
        "sequence_window": ("Sequence window",),
        "gene_name": ("Gene names",),
        "protein_name": ("Protein names",),
    }
    frame = _rename_with_candidates(df.copy(), aliases, required=("uniprot_ids", "positions"))

    before = len(frame)
    frame["uniprot_ids"] = frame["uniprot_ids"].apply(split_uniprot_ids)
    unmapped_rows: list[dict[str, Any]] = []
    frame = _drop_empty_ids(frame, source_file, "phospho_sites", unmapped_rows)

    frame["positions"] = frame["positions"].apply(parse_positions)
    frame = _drop_empty_positions(frame, source_file, "phospho_sites", unmapped_rows)

    frame["amino_acid"] = frame.get("amino_acid", pd.Series(index=frame.index)).apply(
        _normalize_amino_acid(_PHOS_ALLOWED)
    )
    frame["sequence_window"] = frame.get("sequence_window", pd.Series(index=frame.index)).apply(clean_sequence_window)

    _ensure_columns(frame, ("gene_name", "protein_name"))

    frame = frame.explode("positions", ignore_index=True)
    frame.rename(columns={"positions": "position"}, inplace=True)
    frame["position"] = frame["position"].astype(int)

    frame["primary_uniprot"] = frame["uniprot_ids"].apply(_derive_primary_uniprot)
    frame, malformed = _drop_null_primary(frame, source_file, "phospho_sites")
    unmapped_rows.extend(malformed)

    frame["source_file"] = source_file

    ordered = [
        "uniprot_ids",
        "primary_uniprot",
        "position",
        "amino_acid",
        "sequence_window",
        "gene_name",
        "protein_name",
        "source_file",
    ]
    ordered += [col for col in frame.columns if col not in ordered]

    frame = frame[ordered].drop_duplicates(subset=["primary_uniprot", "position", "amino_acid"]).reset_index(drop=True)
    frame.insert(0, "site_id", [f"PH{i:06d}" for i in range(1, len(frame) + 1)])
    logger.info("Parsed phospho_sites: %d rows -> %d rows", before, len(frame))

    report = _build_unmapped_report(unmapped_rows)
    return ParseResult(records=frame, unmapped=report)


def _rename_with_candidates(
    df: pd.DataFrame,
    aliases: dict[str, Sequence[str]],
    required: Iterable[str],
) -> pd.DataFrame:
    rename_map: dict[str, str] = {}
    missing: list[str] = []
    for canonical, candidates in aliases.items():
        match = next((col for col in candidates if col in df.columns), None)
        if match is not None:
            rename_map[match] = canonical
        elif canonical in required:
            missing.append(canonical)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    return df.rename(columns=rename_map, errors="ignore")


def _drop_empty_ids(
    frame: pd.DataFrame,
    source_file: str,
    table: str,
    unmapped_rows: list[dict[str, Any]],
) -> pd.DataFrame:
    mask = frame["uniprot_ids"].map(bool)
    dropped = (~mask).sum()
    if dropped:
        logger.warning("Dropping %d %s rows with empty UniProt IDs", dropped, table)
        _append_unmapped(unmapped_rows, frame[~mask], source_file, "EMPTY_ID")
    return frame[mask].copy()


def _drop_empty_positions(
    frame: pd.DataFrame,
    source_file: str,
    table: str,
    unmapped_rows: list[dict[str, Any]],
) -> pd.DataFrame:
    mask = frame["positions"].map(bool)
    dropped = (~mask).sum()
    if dropped:
        logger.warning("Dropping %d %s rows without numeric positions", dropped, table)
        _append_unmapped(unmapped_rows, frame[~mask], source_file, "NO_POSITION_FOR_SITE")
    return frame[mask].copy()


def _derive_primary_uniprot(ids: list[str]) -> str | None:
    if not ids:
        return None
    first = ids[0]
    return first.split("-", 1)[0]


def _drop_null_primary(
    frame: pd.DataFrame,
    source_file: str,
    table: str,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    mask = frame["primary_uniprot"].notna() & frame["primary_uniprot"].astype(bool)
    dropped = (~mask).sum()
    unmapped: list[dict[str, Any]] = []
    if dropped:
        logger.warning("Dropping %d %s rows with malformed UniProt IDs", dropped, table)
        unmapped = _collect_rows(frame[~mask], source_file, "MALFORMED_ID")
    return frame[mask].copy(), unmapped


def _ensure_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    for column in columns:
        if column not in frame.columns:
            frame[column] = None


def _normalize_amino_acid(allowed: set[str]):
    def _inner(value: Any) -> str | None:
        if pd.isna(value) or value == "":
            return None
        amino = str(value).strip().upper()
        if amino and amino not in allowed:
            logger.warning("Encountered amino acid '%s' not in allowed set %s", amino, sorted(allowed))
        return amino or None

    return _inner


def _append_unmapped(
    accumulator: list[dict[str, Any]],
    rows: pd.DataFrame,
    source_file: str,
    reason: str,
) -> None:
    if reason not in _UNMAPPED_REASONS:
        logger.warning("Unknown unmapped reason '%s'; skipping report entry", reason)
        return
    accumulator.extend(_collect_rows(rows, source_file, reason))


def _collect_rows(rows: pd.DataFrame, source_file: str, reason: str) -> list[dict[str, Any]]:
    collected: list[dict[str, Any]] = []
    for index, record in rows.iterrows():
        collected.append(
            {
                "reason": reason,
                "source_file": source_file,
                "original_index": index,
            }
        )
    return collected


def _build_unmapped_report(entries: list[dict[str, Any]]) -> pd.DataFrame:
    if not entries:
        return pd.DataFrame(columns=["reason", "source_file", "original_index"])
    return pd.DataFrame(entries, columns=["reason", "source_file", "original_index"])


__all__ = [
    "ParseResult",
    "parse_protein_groups",
    "parse_hexnac_sites",
    "parse_phospho_sites",
]
