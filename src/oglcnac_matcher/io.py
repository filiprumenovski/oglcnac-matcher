from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterable, Literal

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CACHE_DIR = DATA_DIR / "processed"

ODB_API_BASE = "https://www.oglcnac.mcw.edu/api/v1"

logger = logging.getLogger(__name__)

_SAFE_NAME_PATTERN = re.compile(r"[^a-z0-9]+")

_PROTEIN_GROUPS_REQUIRED = [
    "Protein IDs",
    "Gene names",
    "Protein names",
]

_HEXNAC_REQUIRED = [
    "Proteins",
    "Leading proteins",
    "Positions within proteins",
    "Amino acid",
    "Sequence window",
]

_HEXNAC_PREFIXES = ("Intensity ",)

_PHOSPHO_REQUIRED = [
    "Proteins",
    "Positions within proteins",
    "Amino acid",
    "Sequence window",
]

_PHOSPHO_PREFIXES = ("Intensity ", "MS/MS Count ")

_MAXQUANT_SHEETS = {
    "protein_groups": {
        "sheet": "proteinGroups",
        "required": _PROTEIN_GROUPS_REQUIRED,
        "prefixes": (),
        "allow_missing": False,
    },
    "hexnac_sites": {
        "sheet": "HexNAc (ST)Sites",
        "required": _HEXNAC_REQUIRED,
        "prefixes": _HEXNAC_PREFIXES,
        "allow_missing": False,
    },
    "phospho_sites": {
        "sheet": "Phospho (STY)Sites",
        "required": _PHOSPHO_REQUIRED,
        "prefixes": _PHOSPHO_PREFIXES,
        "allow_missing": True,
    },
}

FILE_TO_SHEET = {
    "protein-groups.xlsx": ("protein_groups", "proteinGroups"),
    "HexNAc-sites.xlsx": ("hexnac_sites", "HexNAc (ST)Sites"),
    "phospho-sites.xlsx": ("phospho_sites", "Phospho (STY)Sites"),
}


def ensure_dir(path: Path) -> None:
    """Create *path* (and parents) when missing.

    Examples:
        >>> ensure_dir(Path("data/processed"))
    """

    path.mkdir(parents=True, exist_ok=True)


def load_maxquant_excel(
    path: str | Path,
    missing_ok: bool = True,
) -> dict[str, pd.DataFrame]:
    """Load a MaxQuant sheet from its dedicated workbook with validation.

    Parameters:
        path: File system path to one of the accepted MaxQuant Excel exports.
        missing_ok: Allow the phospho file to be absent.

    Examples:
        >>> tables = load_maxquant_excel("in/protein-groups.xlsx")
        >>> tables["protein_groups"].shape
        (1284, 150)

    Returns:
        Dictionary with a single entry keyed by the canonical sheet identifier.

    Raises:
        FileNotFoundError: When the Excel file is missing.
        ValueError: When the filename is not accepted or required columns are missing.
    """

    workbook_path = Path(path)
    filename = workbook_path.name
    if filename not in FILE_TO_SHEET:
        allowed = ", ".join(sorted(FILE_TO_SHEET))
        raise ValueError(
            f"Unrecognized MaxQuant file '{filename}'. Expected one of: {allowed}."
        )

    key, sheet_name = FILE_TO_SHEET[filename]
    spec = _MAXQUANT_SHEETS[key]

    if not workbook_path.exists():
        if spec["allow_missing"] and missing_ok:
            logger.info(
                "Optional MaxQuant file %s missing; returning placeholder for '%s'.",
                workbook_path,
                key,
            )
            return {key: _empty_required_frame(spec["required"])}
        raise FileNotFoundError(
            f"MaxQuant workbook not found at {workbook_path}. Confirm the export path."
        )

    logger.info(
        "Loading MaxQuant workbook %s as key '%s' (sheet '%s').",
        workbook_path,
        key,
        sheet_name,
    )

    try:
        frame = pd.read_excel(workbook_path, sheet_name=sheet_name)
    except ValueError as exc:
        raise ValueError(
            f"Sheet '{sheet_name}' not found in {filename}. "
            "Verify the export configuration."
        ) from exc

    _validate_sheet_frame(
        frame,
        sheet_name,
        spec["required"],
        tuple(spec["prefixes"]),
    )
    return {key: frame}


def save_tables(
    tables: dict[str, pd.DataFrame],
    outdir: str | Path = CACHE_DIR,
    fmt: Literal["csv", "parquet"] = "csv",
) -> dict[str, Path]:
    """Persist DataFrames to disk with predictable names.

    Parameters:
        tables: Mapping of friendly names to DataFrames.
        outdir: Destination directory, created if missing.
        fmt: Output format, ``csv`` or ``parquet``.

    Examples:
        >>> save_tables({"protein_groups": pd.DataFrame()}, "out")
        {'protein_groups': Path('out/protein_groups.csv')}

    Returns:
        Mapping of keys to written file paths.

    Raises:
        ValueError: When an unsupported format is requested.
    """

    output_dir = Path(outdir)
    ensure_dir(output_dir)

    fmt = fmt.lower()
    if fmt not in {"csv", "parquet"}:
        raise ValueError(f"Unsupported format '{fmt}'. Choose 'csv' or 'parquet'.")

    written: dict[str, Path] = {}
    for name, frame in tables.items():
        safe = _safe_name(name)
        destination = output_dir / f"{safe}.{fmt}"
        if fmt == "csv":
            frame.to_csv(destination, index=False, encoding="utf-8")
        else:
            _ensure_pyarrow()
            frame.to_parquet(destination, index=False, engine="pyarrow")

        logger.info("Saved %s rows to %s", len(frame), destination)
        written[name] = destination

    return written


def fetch_odb(
    species: Literal["human", "mouse"],
    force: bool = False,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Fetch O-GlcNAc Database proteins and cache the result.

    Parameters:
        species: Species identifier, ``'human'`` or ``'mouse'``.
        force: Ignore cache and fetch from the API.
        session: Optional preconfigured requests session.

    Examples:
        >>> df = fetch_odb("human")
        >>> df.head(1).columns.tolist()[0]
        'uniprot_id'

    Returns:
        DataFrame containing the API response.

    Raises:
        requests.HTTPError: When the API responds with an error status.
        ValueError: When an unsupported species is requested.
    """

    if species not in {"human", "mouse"}:
        raise ValueError("Species must be 'human' or 'mouse'.")

    cache_path = CACHE_DIR / f"odb_{species}.parquet"
    if cache_path.exists() and not force:
        logger.info("Loading cached ODB data for %s from %s", species, cache_path)
        _ensure_pyarrow()
        return pd.read_parquet(cache_path)

    ensure_dir(cache_path.parent)
    url = f"{ODB_API_BASE}/proteins"
    params = {"species": species}
    logger.info("Fetching ODB data for %s from %s", species, url)

    owns_session = session is None
    active_session = session or requests.Session()

    try:
        response = active_session.get(url, params=params, timeout=30)
        response.raise_for_status()
    finally:
        if owns_session:
            active_session.close()

    payload = response.json()
    if isinstance(payload, dict) and "data" in payload:
        data = payload["data"]
    else:
        data = payload

    if isinstance(data, list):
        frame = pd.json_normalize(data)
    else:
        frame = pd.json_normalize(payload)

    _ensure_pyarrow()
    frame.to_parquet(cache_path, index=False, engine="pyarrow")
    logger.info("Cached ODB data for %s at %s", species, cache_path)

    return frame


def load_oglcnac_atlas(
    species: Literal["human", "mouse"],
    src: str | Path | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """Load and cache the O-GlcNAc Atlas reference table.

    Parameters:
        species: Species identifier, ``'human'`` or ``'mouse'``.
        src: Optional local CSV path to seed the cache.
        force: Ignore existing cache and reread the CSV source.

    Examples:
        >>> df = load_oglcnac_atlas("human", "atlas_human.csv")
        >>> len(df)
        0

    Returns:
        DataFrame sourced from the cache or the provided CSV.

    Raises:
        FileNotFoundError: When *src* is provided but missing.
        ValueError: When no cache exists and no source path is supplied.
    """

    if species not in {"human", "mouse"}:
        raise ValueError("Species must be 'human' or 'mouse'.")

    cache_path = CACHE_DIR / f"atlas_{species}.parquet"

    if src is not None:
        csv_path = Path(src)
        if not csv_path.exists():
            raise FileNotFoundError(f"Atlas CSV not found at {csv_path}.")
        logger.info("Loading Atlas CSV for %s from %s", species, csv_path)
        frame = pd.read_csv(csv_path)
        ensure_dir(cache_path.parent)
        _ensure_pyarrow()
        frame.to_parquet(cache_path, index=False, engine="pyarrow")
        logger.info("Cached Atlas data for %s at %s", species, cache_path)
        return frame

    if cache_path.exists() and not force:
        logger.info("Loading cached Atlas data for %s from %s", species, cache_path)
        _ensure_pyarrow()
        return pd.read_parquet(cache_path)

    raise ValueError("Provide `src` path for local Atlas CSV until remote download is added.")


def _safe_name(name: str) -> str:
    """Convert *name* into a filesystem-friendly token."""

    cleaned = _SAFE_NAME_PATTERN.sub("_", name.lower()).strip("_")
    return cleaned or "table"


def _validate_sheet_frame(
    frame: pd.DataFrame,
    sheet_name: str,
    required_cols: Iterable[str],
    required_prefixes: Iterable[str] = (),
) -> None:
    """Validate presence of required columns in *frame*."""

    missing_cols = [col for col in required_cols if col not in frame.columns]
    if missing_cols:
        raise ValueError(
            f"Sheet '{sheet_name}' is missing columns: {', '.join(missing_cols)}. "
            "Check the MaxQuant export settings."
        )

    missing_prefixes = [
        prefix for prefix in required_prefixes if not _collect_dynamic_columns(frame.columns, (prefix,))
    ]
    if missing_prefixes:
        prefix_msg = ", ".join(missing_prefixes)
        logger.warning(
            "Sheet '%s' missing auxiliary measurement columns (%s) — continuing.",
            sheet_name,
            prefix_msg,
        )


def _empty_required_frame(required_cols: Iterable[str]) -> pd.DataFrame:
    """Create an empty DataFrame with the provided columns."""

    return pd.DataFrame(columns=list(required_cols))


def _collect_dynamic_columns(
    cols: Iterable[str],
    prefixes: Iterable[str],
) -> list[str]:
    """Return columns whose names start with any of *prefixes*."""

    return sorted(
        {col for col in cols for prefix in prefixes if col.startswith(prefix)}
    )


def _ensure_pyarrow() -> None:
    """Ensure that the optional ``pyarrow`` dependency is available."""

    try:
        import pyarrow  # type: ignore  # noqa: F401
    except ImportError as exc:  # pragma: no cover - configuration issue
        raise ImportError(
            "Parquet support requires the 'pyarrow' package. Install it or choose "
            "a CSV workflow."
        ) from exc


__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR",
    "RAW_DIR",
    "CACHE_DIR",
    "ensure_dir",
    "load_maxquant_excel",
    "save_tables",
    "fetch_odb",
    "load_oglcnac_atlas",
]
