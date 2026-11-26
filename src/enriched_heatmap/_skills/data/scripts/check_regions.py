"""Check a region table before building heatmap bins from it.

Reports missing columns, strand encoding, duplicate identifiers, chromosome
names absent from a genome table, windows running off chromosome ends, and the
size of the resulting bin table.

    python check_regions.py peaks.bed --upstream 1000 --downstream 1000 --bin-size 20 \
        --genome chrom.sizes
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

BED_COLS = ["chrom", "start", "end", "name", "score", "strand"]
BED_EXTS = (".bed", ".narrowpeak", ".broadpeak", ".gappedpeak")
KNOWN_STRANDS = {"+", "-", "1", "-1", "+1", "1.0", "-1.0", "+1.0", "0", "0.0", ".", "*"}


def _is_bed_like(path: Path) -> bool:
    lower = path.name.lower()
    if lower.endswith(".gz"):
        lower = lower[:-3]
    return lower.endswith(BED_EXTS)


def read_table(path: Path) -> pd.DataFrame:
    lower = path.name.lower()
    if lower.endswith(".parquet"):
        return pd.read_parquet(path)
    if _is_bed_like(path):
        df = pd.read_csv(path, sep="\t", header=None, comment="#")
        extra = [f"col{i}" for i in range(max(0, df.shape[1] - len(BED_COLS)))]
        df.columns = pd.Index(BED_COLS[: df.shape[1]] + extra)
        return df
    sep = "," if lower.endswith((".csv", ".csv.gz")) else "\t"
    return pd.read_csv(path, sep=sep)


def read_genome(path: Path) -> dict[str, int]:
    g = pd.read_csv(path, sep="\t", header=None, comment="#")
    last_col = g.columns[-1]
    return dict(zip(g[0].astype(str), g[last_col].astype(int), strict=True))


def _chrom_in_genome(chrom: str, genome: dict[str, int]) -> bool:
    alt = chrom[3:] if chrom.startswith("chr") else f"chr{chrom}"
    return chrom in genome or alt in genome


def _lookup_genome_size(chrom: str, genome: dict[str, int]) -> float:
    if chrom in genome:
        return float(genome[chrom])
    alt = chrom[3:] if chrom.startswith("chr") else f"chr{chrom}"
    return float(genome[alt]) if alt in genome else float("nan")


def check(
    df: pd.DataFrame,
    upstream: int,
    downstream: int,
    bin_size: int,
    genome: dict[str, int] | None,
    id_col: str | None,
) -> list[str]:
    msgs: list[str] = []
    missing = [c for c in ("chrom", "start", "end") if c not in df.columns]
    if missing:
        return [f"ERROR missing columns {missing}; have {list(df.columns)[:10]}"]
    if bin_size <= 0:
        return [f"ERROR bin_size must be positive, got {bin_size}"]
    if upstream < 0 or downstream < 0:
        return [
            f"ERROR upstream ({upstream}) and downstream ({downstream}) must be non-negative"
        ]
    if upstream % bin_size or downstream % bin_size:
        msgs.append("ERROR upstream/downstream are not multiples of bin_size")
    n_bins = upstream // bin_size + downstream // bin_size
    if n_bins <= 0:
        msgs.append("ERROR window has 0 bins: upstream + downstream must be positive")

    if id_col is not None:
        if id_col not in df.columns:
            msgs.append(
                f"ERROR id column {id_col!r} missing from table; have {list(df.columns)[:10]}"
            )
        else:
            ids = df[id_col]
            if bool(ids.isna().any()):
                msgs.append(f"ERROR {int(ids.isna().sum())} missing (NaN) region identifiers")
            if not ids.is_unique:
                msgs.append(
                    f"ERROR {int(ids.duplicated().sum())} duplicate region identifiers; "
                    "reset_index() or pass id=<unique column>"
                )
    else:
        idx_s = df.index.to_series()
        if bool(idx_s.isna().any()):
            msgs.append(f"ERROR {int(idx_s.isna().sum())} missing (NaN) region identifiers")
        if not df.index.is_unique:
            msgs.append(
                f"ERROR {int(df.index.duplicated().sum())} duplicate region identifiers; "
                "reset_index() or pass id=<unique column>"
            )

    chrom_raw = df["chrom"]
    chrom_na = chrom_raw.isna() | (chrom_raw.astype(str).str.strip() == "")
    if bool(chrom_na.any()):
        msgs.append(f"ERROR {int(chrom_na.sum())} missing values in 'chrom'; drop those rows first")

    start_num = pd.to_numeric(df["start"], errors="coerce")
    end_num = pd.to_numeric(df["end"], errors="coerce")
    for col_name, raw_col, num_col in (
        ("start", df["start"], start_num),
        ("end", df["end"], end_num),
    ):
        n_na = int(raw_col.isna().sum())
        if n_na > 0:
            msgs.append(f"ERROR {n_na} missing values in {col_name!r}; drop those rows first")
        n_bad = int((num_col.isna() & ~raw_col.isna()).sum())
        if n_bad > 0:
            msgs.append(f"ERROR {n_bad} non-numeric values in {col_name!r}")

    valid_coords = (~chrom_na) & start_num.notna() & end_num.notna()
    if bool(valid_coords.any()):
        v_start = start_num.loc[valid_coords]
        v_end = end_num.loc[valid_coords]
        if bool((v_end < v_start).any()):
            msgs.append("ERROR rows with end < start")
        if int((v_start == 1).sum()) > 0 and int((v_start == 0).sum()) == 0:
            msgs.append("WARN starts at 1 but never 0: coordinates may be 1-based")

    if "strand" in df.columns:
        non_na_strands = df["strand"].dropna().astype(str).str.strip()
        vals = {v for v in non_na_strands.tolist() if v != ""}
        if vals <= KNOWN_STRANDS:
            msgs.append(f"OK strand values {sorted(vals)}")
        else:
            msgs.append(
                f"WARN unrecognised strand values {sorted(vals - KNOWN_STRANDS)} (read as +)"
            )
    else:
        msgs.append("INFO no strand column: all regions read as +")

    if genome is not None:
        valid_chroms = {str(c).strip() for c in df.loc[~chrom_na, "chrom"].tolist()}
        bad = sorted(c for c in valid_chroms if not _chrom_in_genome(c, genome))
        if bad:
            msgs.append(f"ERROR chromosomes not in genome (eh.bins raises): {bad[:10]}")
        if bool(valid_coords.any()):
            v_chrom = df.loc[valid_coords, "chrom"].astype(str).str.strip()
            v_mid = (start_num.loc[valid_coords] + end_num.loc[valid_coords]) // 2
            g_map = genome
            v_size = v_chrom.map(lambda c: _lookup_genome_size(c, g_map))
            flank = max(upstream, downstream)
            off = int(((v_mid - flank < 0) | (v_mid + flank > v_size)).sum())
            if off:
                msgs.append(
                    f"INFO {off} centred windows touch a chromosome edge; those bins are NaN"
                )
    else:
        msgs.append(
            "INFO no genome given: pass genome= to eh.bins or pymisha errors past chromosome ends"
        )
    msgs.append(f"OK {len(df)} regions x {n_bins} bins = {len(df) * n_bins:,} bin rows")
    return msgs


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("table", type=Path)
    p.add_argument("--upstream", type=int, default=1000)
    p.add_argument("--downstream", type=int, default=1000)
    p.add_argument("--bin-size", type=int, default=20)
    p.add_argument("--genome", type=Path, help="chrom.sizes (chrom<TAB>size)")
    p.add_argument("--id", help="column holding region identifiers")
    a = p.parse_args(list(argv) if argv is not None else None)
    msgs = check(
        read_table(a.table),
        a.upstream,
        a.downstream,
        a.bin_size,
        read_genome(a.genome) if a.genome else None,
        a.id,
    )
    print("\n".join(msgs))
    return 1 if any(m.startswith("ERROR") for m in msgs) else 0


if __name__ == "__main__":
    sys.exit(main())
