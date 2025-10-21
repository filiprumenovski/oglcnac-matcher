from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


logger = logging.getLogger(__name__)

_DEFAULT_CONFIG = {
    "matches_dir": "data/processed/matches",
    "outdir": "reports/quick",
    "title": "O-GlcNAc Matcher Quick Report",
    "sections": {
        "counters": [
            {"csv": "proteins_overlap.csv", "label": "Proteins overlap"},
            {"csv": "hexnac_site_matches.csv", "label": "HexNAc sites"},
        ],
        "category_bars": [
            {
                "csv": "proteins_overlap.csv",
                "column": "match_level_protein",
                "title": "Protein match tiers",
                "basename": "protein_match_tiers",
            },
            {
                "csv": "hexnac_site_matches.csv",
                "column": "site_match_tier",
                "title": "Site match tiers",
                "basename": "site_match_tiers",
            },
        ],
        "venn": [
            {
                "title": "Matched ODB vs HexNAc Sites vs Phospho",
                "basename": "venn_matched_vs_sites_vs_phospho",
                "kind": "derived_match_sets"
            }
        ],
    },
}

_SPLIT_PATTERN = re.compile(r"[;,]")


def ensure_outdir(path: str | Path) -> Path:
    """Ensure that the output directory exists."""

    outdir = Path(path)
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def load_csv(path: Path) -> pd.DataFrame:
    """Load a CSV file, returning an empty DataFrame on failure."""

    if not path.exists():
        logger.warning("CSV not found: %s", path)
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        logger.warning("CSV empty: %s", path)
        return pd.DataFrame()
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Unable to load %s: %s", path, exc)
        return pd.DataFrame()

    logger.info("Loaded %s (%d rows, %d columns)", path, len(df), len(df.columns))
    return df


def count_column(df: pd.DataFrame, column: str, top_n: int | None) -> pd.DataFrame:
    """Return category counts for the requested column."""

    if df.empty:
        logger.warning("Cannot count column '%s': dataframe is empty", column)
        return pd.DataFrame(columns=["category", "count"])
    if column not in df.columns:
        logger.warning("Column '%s' not present; available columns: %s", column, list(df.columns))
        return pd.DataFrame(columns=["category", "count"])

    series = df[column].fillna("(missing)").astype(str).str.strip()
    series = series.replace("", "(missing)")
    counts = (
        series.value_counts(dropna=False)
        .rename_axis("category")
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )
    if top_n is not None:
        counts = counts.head(top_n)

    logger.info("Counted %d categories for '%s'", len(counts), column)
    return counts


def plot_bar(counts: pd.DataFrame, title: str, out_png: Path) -> None:
    """Render a categorical bar plot and save to PNG."""

    if counts.empty:
        raise ValueError("Counts dataframe is empty; nothing to plot")

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(9, 5), dpi=200)
    sns.barplot(data=counts, x="category", y="count", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Category")
    ax.set_ylabel("Count")
    rotation = 45 if len(counts) > 6 else 0
    # Use plt.setp to avoid set_xticklabels warning
    plt.setp(ax.get_xticklabels(), rotation=rotation, ha="right")
    if len(counts) <= 20:
        # Robust annotation across versions without relying on containers
        for p in ax.patches:
            h = p.get_height()
            ax.annotate(
                f"{int(h)}",
                (p.get_x() + p.get_width() / 2, h),
                ha="center",
                va="bottom",
                xytext=(0, 3),
                textcoords="offset points",
            )
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    logger.info("Wrote bar chart: %s", out_png)
    plt.close(fig)


def plot_venn(sets: list[tuple[str, set[str]]], title: str, out_png: Path) -> None:
    """Render a 2- or 3-set Venn diagram to PNG."""
    try:
        from matplotlib_venn import venn2, venn3  # type: ignore
    except Exception:
        logger.warning("matplotlib-venn is not installed; skipping Venn diagram '%s'", title)
        return

    if len(sets) not in (2, 3):
        raise ValueError("Venn diagrams support only 2 or 3 sets")

    fig, ax = plt.subplots(figsize=(6, 6), dpi=200)
    names = [name for name, _ in sets]
    values = [value for _, value in sets]

    if len(values) == 2:
        venn2(subsets=values, set_labels=names, ax=ax)
    else:
        venn3(subsets=values, set_labels=names, ax=ax)

    ax.set_title(title)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    logger.info("Wrote Venn diagram: %s", out_png)
    plt.close(fig)


def _build_derived_match_sets(matches_dir: Path) -> list[tuple[str, set[str]]]:
    """
    Build 3 sets from matches outputs only:
      A: all ODB-matched proteins (proteins_overlap.csv → 'odb_uniprot')
      B: proteins with ≥1 matched HexNAc site (hexnac_site_matches.csv where site_match_tier != 'NO_MATCH' → 'odb_uniprot')
      C: proteins with phospho overlap on matched ODB proteins (phospho_on_odb_proteins.csv → 'odb_uniprot')
    Missing files yield empty sets; function never raises.
    """
    def _colset(csv_name: str, col: str, predicate: callable | None = None) -> set[str]:
        p = _resolve_path(matches_dir, csv_name)
        if not p.exists():
            logger.warning("CSV not found for derived sets: %s", p)
            return set()
        try:
            df = pd.read_csv(p)
        except Exception as e:
            logger.warning("Failed reading %s: %s", p, e)
            return set()
        if predicate is not None:
            try:
                df = df[predicate(df)]
            except Exception as e:
                logger.warning("Predicate failed for %s: %s", p, e)
                return set()
        if col not in df.columns:
            logger.warning("Column '%s' missing in %s", col, p)
            return set()
        vals = {str(x).strip().upper() for x in df[col].dropna().unique().tolist()}
        return {v for v in vals if v}

    set_matched = _colset("proteins_overlap.csv", "odb_uniprot")
    set_sites   = _colset("hexnac_site_matches.csv", "odb_uniprot", predicate=lambda d: d["site_match_tier"].astype(str) != "NO_MATCH")
    set_phospho = _colset("phospho_on_odb_proteins.csv", "odb_uniprot")
    return [("ODB matched", set_matched), ("HexNAc sites matched", set_sites), ("Phospho overlap", set_phospho)]


def render_report(config_or_path: Path | dict[str, Any]) -> int:
    """Render the quick Markdown report from a configuration JSON or dict config."""

    if isinstance(config_or_path, Path):
        config_path = config_or_path
        if not config_path.exists():
            logger.error("Config not found: %s", config_path)
            return 1
        try:
            config: dict[str, Any] = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            logger.error("Invalid JSON in %s: %s", config_path, exc)
            return 1
    else:
        # Assume dict-like config already provided
        config = config_or_path

    matches_dir = Path(config.get("matches_dir", "."))
    outdir = ensure_outdir(config.get("outdir", "reports/quick"))
    title = config.get("title", "O-GlcNAc Matcher Quick Report")
    sections = config.get("sections", {})

    warnings: list[str] = []
    counters_rows: list[tuple[str, str]] = []
    bar_outputs: list[tuple[str, str]] = []
    venn_outputs: list[tuple[str, str]] = []

    for counter in sections.get("counters", []):
        label = counter.get("label", "Unnamed")
        csv_name = counter.get("csv")
        if not csv_name:
            warnings.append(f"Counter '{label}' missing CSV path; skipped.")
            continue
        csv_path = _resolve_path(matches_dir, csv_name)
        df = load_csv(csv_path)
        if df.empty:
            warnings.append(f"Counter '{label}' skipped: no data in {csv_path}.")
            counters_rows.append((label, "(missing)"))
            continue
        counters_rows.append((label, str(len(df))))

    for bar in sections.get("category_bars", []):
        csv_name = bar.get("csv")
        column = bar.get("column")
        title_bar = bar.get("title", "Category Counts")
        basename = bar.get("basename", _sanitize_basename(title_bar))
        top_n = bar.get("top_n")
        if top_n is not None:
            try:
                top_n = int(top_n)
            except (TypeError, ValueError):
                warnings.append(f"Bar chart '{title_bar}' has non-integer top_n; ignoring limit.")
                top_n = None
        if not csv_name or not column:
            warnings.append(f"Bar chart '{title_bar}' missing csv or column; skipped.")
            continue
        csv_path = _resolve_path(matches_dir, csv_name)
        df = load_csv(csv_path)
        counts = count_column(df, column, top_n)
        if counts.empty:
            warnings.append(f"Bar chart '{title_bar}' skipped: unable to derive counts from {csv_path}.")
            continue
        out_png = outdir / f"{basename}.png"
        try:
            plot_bar(counts, title_bar, out_png)
            bar_outputs.append((title_bar, out_png.name))
        except Exception as exc:  # pragma: no cover - defensive
            warnings.append(f"Bar chart '{title_bar}' failed: {exc}.")

    for spec in sections.get("venn", []):
        title_venn = spec.get("title", "Venn Diagram")
        basename = spec.get("basename", _sanitize_basename(title_venn))
        out_png = outdir / f"{basename}.png"

        # New: derived sets from matches/ only
        if spec.get("kind") == "derived_match_sets":
            sets = _build_derived_match_sets(matches_dir)
            if all(len(s[1]) == 0 for s in sets):
                warnings.append(f"Venn '{title_venn}' skipped: all derived sets empty.")
                continue
            try:
                plot_venn(sets, title_venn, out_png)
                venn_outputs.append((title_venn, out_png.name))
            except Exception as exc:  # pragma: no cover - defensive
                warnings.append(f"Venn '{title_venn}' failed: {exc}.")
            continue

        # Legacy explicit CSV-based sets
        set_specs = spec.get("sets", [])
        if len(set_specs) not in (2, 3):
            warnings.append(f"Venn '{title_venn}' skipped: requires 2 or 3 sets.")
            continue

        sets_data: list[tuple[str, set[str]]] = []
        skip = False
        for entry in set_specs:
            name = entry.get("name", "Unnamed")
            csv_name = entry.get("csv")
            column = entry.get("id_column")
            if not csv_name or not column:
                warnings.append(f"Venn '{title_venn}' skipped: set '{name}' missing csv or id_column.")
                skip = True
                break
            csv_path = _resolve_path(matches_dir, csv_name)
            df = load_csv(csv_path)
            if df.empty:
                warnings.append(f"Venn '{title_venn}' skipped: no data for set '{name}' ({csv_path}).")
                skip = True
                break
            if column not in df.columns:
                warnings.append(f"Venn '{title_venn}' skipped: column '{column}' missing in {csv_path}.")
                skip = True
                break
            ids = _extract_ids(df[column])
            if not ids:
                warnings.append(f"Venn '{title_venn}' skipped: set '{name}' has no identifiers in column '{column}'.")
                skip = True
                break
            sets_data.append((name, ids))

        if skip:
            continue

        try:
            plot_venn(sets_data, title_venn, out_png)
            venn_outputs.append((title_venn, out_png.name))
        except Exception as exc:  # pragma: no cover - defensive
            warnings.append(f"Venn '{title_venn}' failed: {exc}.")

    summary_path = outdir / "summary.md"
    summary_path.write_text(_build_markdown(title, counters_rows, bar_outputs, venn_outputs, warnings), encoding="utf-8")
    logger.info("Wrote Markdown summary: %s", summary_path)

    return 0


def _build_markdown(
    title: str,
    counters: list[tuple[str, str]],
    bars: list[tuple[str, str]],
    venns: list[tuple[str, str]],
    warnings: list[str],
) -> str:
    lines: list[str] = [f"# {title}", ""]

    lines.append("## Counters")
    if counters:
        lines.append("| Item | Rows |")
        lines.append("|---|---|")
        for label, count in counters:
            lines.append(f"| {label} | {count} |")
    else:
        lines.append("No counters available.")
    lines.append("")

    lines.append("## Category Bars")
    if bars:
        for title_bar, filename in bars:
            lines.append(f"- **{title_bar}**  ")
            lines.append(f"![{filename}](./{filename})")
            lines.append("")
    else:
        lines.append("No category charts available.")
        lines.append("")

    lines.append("## Venn Diagrams")
    if venns:
        for title_venn, filename in venns:
            lines.append(f"- **{title_venn}**  ")
            lines.append(f"![{filename}](./{filename})")
            lines.append("")
    else:
        lines.append("No Venn diagrams available.")
        lines.append("")

    if warnings:
        lines.append("## Warnings")
        for warning in warnings:
            lines.append(f"- {warning}")
        lines.append("")

    return "\n".join(lines)


def _resolve_path(base_dir: Path, csv_name: str) -> Path:
    """Resolve a CSV path relative to `base_dir` with smart fallbacks.

    Order of checks:
    1) Absolute path → return as-is
    2) `base_dir / csv_name` (e.g., matches dir)
    3) Common data roots within the repo (processed/raw/external, ODB dirs, project root)
    4) Fallback to `base_dir / csv_name` even if missing (so callers can log a helpful path)
    """
    candidate = Path(csv_name)
    if candidate.is_absolute():
        return candidate

    primary = base_dir / candidate
    if primary.exists():
        return primary

    common_roots = (
        Path("data/processed/matches"),  # typical matcher outputs
        Path("data/processed"),
        Path("data/raw"),
        Path("data/external"),
        Path("data/odb/raw"),           # ODB CSVs often live here
        Path("data/odb"),
        Path("data"),
        Path("."),                      # repo root as last resort
    )
    for root in common_roots:
        alt = root / candidate
        if alt.exists():
            return alt

    return primary


def _sanitize_basename(text: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", text).strip("_")
    return safe or "chart"


def _extract_ids(series: pd.Series) -> set[str]:
    ids: set[str] = set()
    for value in series.dropna():
        if isinstance(value, (list, tuple, set)):
            iterable = value
        else:
            text = str(value)
            if _SPLIT_PATTERN.search(text):
                iterable = _SPLIT_PATTERN.split(text)
            else:
                iterable = [text]
        for token in iterable:
            token_str = str(token).strip()
            if token_str:
                ids.add(token_str)
    return ids


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a quick Markdown summary of matcher outputs.")
    parser.add_argument("--config", type=Path, help="Path to the report configuration JSON.")
    parser.add_argument("--matches-dir", type=Path, help="Override matches_dir in config (e.g., data/processed/matches).")
    parser.add_argument("--outdir", type=Path, help="Override output directory (e.g., reports/quick).")
    parser.add_argument("--title", type=str, help="Override report title.")
    parser.add_argument("--dump-default-config", action="store_true", help="Print the default configuration JSON and exit.")
    parser.add_argument("--quiet", action="store_true", help="Reduce logging verbosity.")
    return parser.parse_args()


def _configure_logging(quiet: bool) -> None:
    level = logging.WARNING if quiet else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(message)s")


def main() -> int:
    """CLI entry point for report generation."""

    args = _parse_args()
    _configure_logging(args.quiet)

    if args.dump_default_config:
        print(json.dumps(_DEFAULT_CONFIG, indent=2))
        return 0

    if not args.config:
        logger.info("No --config provided; using built-in defaults.")
        config: dict[str, Any] = json.loads(json.dumps(_DEFAULT_CONFIG))
    else:
        config = {"__CONFIG_PATH__": str(args.config)}  # marker for debugging
        # Let render_report load/validate; start with path
        cfg_path = args.config
        if not cfg_path.exists():
            logger.error("Config not found: %s", cfg_path)
            return 1
        try:
            config = json.loads(cfg_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            logger.error("Invalid JSON in %s: %s", cfg_path, exc)
            return 1

    # Apply CLI overrides if provided
    if args.matches_dir is not None:
        config["matches_dir"] = str(args.matches_dir)
    if args.outdir is not None:
        config["outdir"] = str(args.outdir)
    if args.title is not None:
        config["title"] = args.title

    return render_report(config)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ensure_outdir",
    "load_csv",
    "count_column",
    "plot_bar",
    "plot_venn",
    "render_report",
    "main",
]
