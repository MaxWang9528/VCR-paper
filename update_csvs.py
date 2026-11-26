"""Bring every CSV in the repo up to date, then redraw the figures.

    python update_csvs.py                    # fetch new days, rerun main.py, save figures
    python update_csvs.py --through 2026-11-25
    python update_csvs.py --dry-run          # only report what the fetch would add
    python update_csvs.py --no-fetch         # rerun the pipeline on the CSVs as they are

Step 1 appends the trading days after the newest row of SPX.csv / VIX.csv, fetched from
Yahoo Finance (yfinance ^GSPC / ^VIX), in the same Investing.com format (quoted fields,
MM/DD/YYYY, newest row first). Existing rows are never rewritten, and the newest existing
closes are re-fetched and compared first, so a wrong series is refused before anything is
written.

Step 2 runs ``main.main()`` unchanged with a non-interactive matplotlib backend. That
rewrites OUT.csv, writes mrperf.csv / vixperf.csv / vcrperf.csv (the series the commented
lines in ``Grapher.all_performance`` used to save), and saves each chart to figures/.
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import sys
import tempfile
from datetime import date, datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
HEADER = '"Date","Price","Open","High","Low","Vol.","Change %"'
TICKERS = {"SPX.csv": "^GSPC", "VIX.csv": "^VIX"}
OVERLAP_ROWS = 5          # newest existing rows re-fetched and compared
OVERLAP_FAIL_REL = 0.01   # refuse the update if a re-fetched close differs by more than 1%

FIGURES = [  # in the order main.main() draws them
    "01_recent_vs_next_realized_scatter.png",
    "02_recent_vs_next_realized_buckets.png",
    "03_vix_vs_next_realized_252d.png",
    "04_avg_difference_vs_avg_vix_sampled.png",
    "05_mrvol_sq_vs_squared_difference_buckets.png",
    "06_vix_decomposition_252d.png",
    "07_vcr_performance_252d.png",
    "08_all_metric_performance_252d.png",
]
FIG_DPI = 110  # 14x8 in -> 1540x880 px, the size of the existing figures


# ------------------------------------------------------------------ step 1: fetch

def _q2(x) -> Decimal:
    return Decimal(repr(float(x))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def _row(d: date, o, h, lo, c, change: str) -> str:
    fields = [d.strftime("%m/%d/%Y"), f"{c:,.2f}", f"{o:,.2f}", f"{h:,.2f}", f"{lo:,.2f}", "", change]
    return ",".join(f'"{f}"' for f in fields)


def _last_completed_session() -> date:
    """Today in New York once the close has settled (after 17:00), otherwise yesterday."""
    now = datetime.now(ZoneInfo("America/New_York"))
    return now.date() if now.hour >= 17 else now.date() - timedelta(days=1)


def _fetch(ticker: str, start: date, end: date):
    import yfinance as yf

    df = yf.Ticker(ticker).history(start=start.isoformat(), end=(end + timedelta(days=1)).isoformat(),
                                   interval="1d", auto_adjust=False, actions=False)
    df = df[["Open", "High", "Low", "Close"]].dropna(subset=["Close"])
    df.index = [ts.date() for ts in df.index.tz_localize(None)]
    return df[(df.index >= start) & (df.index <= end)].sort_index()


def plan_update(path: Path, through: date) -> tuple[str, list[str], list[str]]:
    """-> (new file text, new rows newest-first, warnings). Raises RuntimeError on bad data."""
    text = path.read_text(encoding="utf-8")
    header, _, rest = text.partition("\n")
    if header.rstrip("\r") != HEADER:
        raise RuntimeError(f"{path.name}: unexpected header {header!r}")
    nl = "\r\n" if header.endswith("\r") else "\n"
    rows = [r for r in csv.reader(io.StringIO(rest)) if r]
    dates = [datetime.strptime(r[0], "%m/%d/%Y").date() for r in rows]
    closes = [Decimal(r[1].replace(",", "")) for r in rows]
    last = dates[0]
    if through <= last:
        return text, [], []

    fetched = _fetch(TICKERS[path.name], last - timedelta(days=14), through)
    warnings = []
    for d, c in zip(dates[:OVERLAP_ROWS], closes[:OVERLAP_ROWS]):
        if d not in fetched.index:
            continue
        theirs = float(fetched.at[d, "Close"])
        if abs(theirs - float(c)) / float(c) > OVERLAP_FAIL_REL:
            raise RuntimeError(f"{path.name} {d}: Yahoo close {theirs:.2f} vs file {c}; refusing to update")
        if abs(theirs - float(c)) > 0.015:
            warnings.append(f"{path.name} {d}: Yahoo close {theirs:.2f} vs file {c}")

    new_rows = []
    prev = closes[0]
    for d, r in fetched[fetched.index > last].iterrows():
        vals = [r["Close"] if v != v else v for v in (r["Open"], r["High"], r["Low"], r["Close"])]
        o, h, lo, c = (_q2(v) for v in vals)
        pct = _q2((c / prev - 1) * 100)
        new_rows.append(_row(d, o, h, lo, c, f"{pct if pct != 0 else Decimal('0.00')}%"))
        prev = c
    new_rows.reverse()  # file is newest-first
    return header + "\n" + "".join(r + nl for r in new_rows) + rest, new_rows, warnings


def update_data(through: date, dry_run: bool) -> None:
    plans = {name: plan_update(ROOT / name, through) for name in TICKERS}  # validate both before writing
    for name, (text, new_rows, warnings) in plans.items():
        span = f" ({new_rows[-1][1:11]} .. {new_rows[0][1:11]})" if new_rows else ""
        print(f"{name}: {'would add' if dry_run else 'added'} {len(new_rows)} row(s){span}")
        for w in warnings:
            print(f"  WARNING: {w}")
        if new_rows and not dry_run:
            fd, tmp = tempfile.mkstemp(dir=ROOT, prefix=f".{name}.")
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as fh:
                fh.write(text)
            os.replace(tmp, ROOT / name)


# ------------------------------------------------------------------ step 2: pipeline

def run_pipeline() -> None:
    os.environ["MPLBACKEND"] = "Agg"
    import matplotlib
    matplotlib.use("Agg")
    import mplcursors
    import numpy as np
    from matplotlib import pyplot as plt

    import Grapher
    import main

    mplcursors.cursor = lambda *a, **k: type("NoCursor", (), {"connect": lambda self, *a, **k: (lambda f: f)})()
    Grapher.mplcursors = mplcursors

    (ROOT / "figures").mkdir(exist_ok=True)
    names = iter(FIGURES)

    def save_instead_of_show(*_, **__):
        name = next(names)
        plt.gcf().savefig(ROOT / "figures" / name, dpi=FIG_DPI)
        plt.close("all")
        print(f"saved figures/{name}")

    plt.show = save_instead_of_show

    original_all_performance = Grapher.Grapher.all_performance

    def all_performance(self, days=1):
        # Same series as the commented-out to_csv lines in Grapher.all_performance.
        d = self.data
        realized = d["% Change Recent Volatility"].rolling(window=days).mean()
        perf = {
            "mrperf.csv": ("Average MR Performance", d["MR Adjustment"]),
            "vixperf.csv": ("Average VIX Performance", d["VIX Level"]),
            "vcrperf.csv": ("Average VCR Performance", d["VCR"]),
        }
        for file, (label, col) in perf.items():
            np.abs(col.rolling(window=days).mean() - realized).rename(label).to_csv(ROOT / file, lineterminator="\n")
            print(f"wrote {file}")
        return original_all_performance(self, days=days)

    Grapher.Grapher.all_performance = all_performance

    cwd = os.getcwd()
    os.chdir(ROOT)  # main.py uses relative paths
    try:
        main.main()
    finally:
        os.chdir(cwd)
    # pandas on Windows writes CRLF; keep OUT.csv's committed LF endings so diffs stay readable.
    out = ROOT / "OUT.csv"
    out.write_bytes(out.read_bytes().replace(b"\r\n", b"\n"))


def main_cli() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--through", type=date.fromisoformat, default=None, metavar="YYYY-MM-DD",
                    help="last date to fetch (default: last completed US session)")
    ap.add_argument("--dry-run", action="store_true", help="report what would be fetched; write nothing")
    ap.add_argument("--no-fetch", action="store_true", help="skip fetching; just rerun the pipeline")
    args = ap.parse_args()
    sys.path.insert(0, str(ROOT))

    if not args.no_fetch:
        try:
            update_data(args.through or _last_completed_session(), args.dry_run)
        except RuntimeError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
    if not args.dry_run:
        run_pipeline()
    return 0


if __name__ == "__main__":
    sys.exit(main_cli())
