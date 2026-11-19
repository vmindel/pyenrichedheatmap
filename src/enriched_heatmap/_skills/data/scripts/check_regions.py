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
from pathlib import Path

import numpy as np
import pandas as pd

BED_COLS = ["chrom", "start", "end", "name", "score", "strand"]


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix in (".bed", ".narrowPeak", ".broadPeak") or path.name.endswith(".bed.gz"):
        df = pd.read_csv(path, sep="\t", header=None, comment="#")
        df.columns = BED_COLS[: df.shape[1]] + [f"col{i}" for i in range(df.shape[1] - 6)]
        return df
    sep = "," if path.suffix == ".csv" else "\t"
    return pd.read_csv(path, sep=sep)


def read_genome(path: Path) -> dict[str, int]:
    g = pd.read_csv(path, sep="\t", header=None, comment="#")
    return dict(zip(g[0].astype(str), g[g.columns[-1]].astype(int), strict=True))


def check(df: pd.DataFrame, upstream: int, downstream: int, bin_size: int,
          genome: dict[str, int] | None, id_col: str | None) -> list[str]:
    msgs: list[str] = []
    missing = [c for c in ("chrom", "start", "end") if c not in df.columns]
    if missing:
        return [f"ERROR missing columns {missing}; have {list(df.columns)[:10]}"]
    if upstream % bin_size or downstream % bin_size:
        msgs.append("ERROR upstream/downstream are not multiples of bin_size")
    ids = df[id_col] if id_col else df.index.to_series()
    if not ids.is_unique:
        msgs.append(f"ERROR {ids.duplicated().sum()} duplicate region identifiers; "
                    "reset_index() or pass id=<unique column>")
    if (df["end"] < df["start"]).any():
        msgs.append("ERROR rows with end < start")
    if (df["start"] == 1).sum() > 0 and (df["start"] == 0).sum() == 0:
        msgs.append("WARN starts at 1 but never 0: coordinates may be 1-based")
    if "strand" in df.columns:
        vals = set(df["strand"].astype(str).str.strip())
        known = {"+", "-", "1", "-1", "0", "."}
        msgs.append(f"OK strand values {sorted(vals)}" if vals <= known
                    else f"WARN unrecognised strand values {sorted(vals - known)} (read as +)")
    else:
        msgs.append("INFO no strand column: all regions read as +")
    if genome is not None:
        chroms = set(df["chrom"].astype(str))
        def ok(c):
            return c in genome or (c[3:] if c.startswith("chr") else f"chr{c}") in genome
        bad = sorted(c for c in chroms if not ok(c))
        if bad:
            msgs.append(f"ERROR chromosomes not in genome (eh.bins raises): {bad[:10]}")
        mid = (df["start"] + df["end"]) // 2
        size = df["chrom"].astype(str).map(lambda c: genome.get(c, genome.get(
            c[3:] if c.startswith("chr") else f"chr{c}", np.nan)))
        off = ((mid - max(upstream, downstream) < 0) | (mid + max(upstream, downstream) > size)).sum()
        if off:
            msgs.append(f"INFO {off} centred windows touch a chromosome edge; those bins are NaN")
    else:
        msgs.append("INFO no genome given: pass genome= to eh.bins or pymisha errors past chromosome ends")
    n_bins = upstream // bin_size + downstream // bin_size
    msgs.append(f"OK {len(df)} regions x {n_bins} bins = {len(df) * n_bins:,} bin rows")
    return msgs


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("table", type=Path)
    p.add_argument("--upstream", type=int, default=1000)
    p.add_argument("--downstream", type=int, default=1000)
    p.add_argument("--bin-size", type=int, default=20)
    p.add_argument("--genome", type=Path, help="chrom.sizes (chrom<TAB>size)")
    p.add_argument("--id", help="column holding region identifiers")
    a = p.parse_args(argv)
    msgs = check(read_table(a.table), a.upstream, a.downstream, a.bin_size,
                 read_genome(a.genome) if a.genome else None, a.id)
    print("\n".join(msgs))
    return 1 if any(m.startswith("ERROR") for m in msgs) else 0


if __name__ == "__main__":
    sys.exit(main())
