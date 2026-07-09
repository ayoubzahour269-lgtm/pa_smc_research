#!/usr/bin/env python3
# scripts/freeze_nasdaq.py
# Gele les 4 snapshots Nasdaq (H1/M15 x BID/ASK) en immuable, schema aligne sur l'or.
# Idempotent : refuse d'ecraser un snapshot deja gele (garde-fou immuabilite).

import time, json, hashlib
from datetime import datetime, timezone
from pathlib import Path
from importlib.metadata import version as _pkgver

import pandas as pd
import dukascopy_python
from dukascopy_python import instruments as ins

PROJECT = Path.cwd()
if PROJECT.name in ("scripts", "notebooks"):
    PROJECT = PROJECT.parent
SNAP_DIR = PROJECT / "data" / "snapshots"
SNAP_DIR.mkdir(parents=True, exist_ok=True)

INSTR_NAME = "INSTRUMENT_IDX_AMERICA_E_NQ_100"
INSTR      = getattr(ins, INSTR_NAME)
START, END = datetime(2015, 1, 1), datetime(2026, 7, 1)
VERSION    = "v1"

INTERVALS = {"H1":  dukascopy_python.INTERVAL_HOUR_1,
             "M15": dukascopy_python.INTERVAL_MIN_15}
SIDES     = {"BID": dukascopy_python.OFFER_SIDE_BID,
             "ASK": dukascopy_python.OFFER_SIDE_ASK}
GAP_THRESH = {"H1": pd.Timedelta(hours=1), "M15": pd.Timedelta(minutes=15)}


def fetch_full(interval, side):
    frames = []
    for year in range(START.year, END.year + 1):
        cs = max(datetime(year, 1, 1), START)
        ce = min(datetime(year + 1, 1, 1), END)
        if cs >= ce:
            continue
        frames.append(dukascopy_python.fetch(INSTR, interval, side, cs, ce))
    return pd.concat(frames).sort_index()


def freeze_one(tf, side):
    interval, offer = INTERVALS[tf], SIDES[side]
    base = f"NAS100_{tf}_{side}_2015-01-01_2026-06-30_{VERSION}"
    csv_path      = SNAP_DIR / f"{base}.csv"
    manifest_path = SNAP_DIR / f"{base}.manifest.json"

    if csv_path.exists() or manifest_path.exists():
        print(f"[SKIP] {base} existe deja -- non reecrit (immuable).")
        return base, None

    t0 = time.time()
    df = fetch_full(interval, offer)

    bad = (
        (df["high"] < df[["open", "close"]].max(axis=1)) |
        (df["low"]  > df[["open", "close"]].min(axis=1)) |
        (df["high"] < df["low"])
    )
    gaps = df.index.to_series().diff()
    big  = gaps[gaps > GAP_THRESH[tf]]

    med_by_year = df.groupby(df.index.year).apply(
        lambda g: int(g.groupby(g.index.date).size().median())
    ).to_dict()
    med_by_year = {int(k): int(v) for k, v in med_by_year.items()}

    df.to_csv(csv_path)
    sha = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    gap_key = "gaps_over_1h" if tf == "H1" else "gaps_over_15min"

    manifest = {
        "instrument": "NAS100 - Nasdaq-100 index (CFD Dukascopy)",
        "instrument_constant": INSTR_NAME,
        "instrument_symbol": INSTR,
        "source": "Dukascopy Bank SA (via dukascopy-python)",
        "library_version": _pkgver("dukascopy-python"),
        "offer_side": side,
        "interval": tf,
        "timezone": "UTC",
        "start": str(df.index.min()),
        "end": str(df.index.max()),
        "n_bars": int(len(df)),
        "columns": list(df.columns),
        "price_close_min": round(float(df["close"].min()), 3),
        "price_close_max": round(float(df["close"].max()), 3),
        "nan_count": int(df.isna().sum().sum()),
        "ohlc_violations": int(bad.sum()),
        "volume_nonpos": int((df["volume"] <= 0).sum()),
        gap_key: int(len(big)),
        "largest_gap": str(big.max()),
        "session_median_bars_per_day_by_year": med_by_year,
        "session_note": ("Regime seance courte 2015-2017 (~15 barres/j H1) puis "
                         "quasi-24h des 2019 (~23 barres/j, aligne sur l'or). "
                         "REPLAY des hypotheses restreint a >=2019 pour repere intraday commun."),
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "csv_file": csv_path.name,
        "sha256": sha,
        "version": VERSION,
        "note": "IMMUABLE - ne jamais reecrire. Toute nouvelle donnee = nouvelle version.",
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    check = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    assert len(check) == len(df), "round-trip: n_bars different"
    assert hashlib.sha256(csv_path.read_bytes()).hexdigest() == sha, "sha256 instable"

    print(f"[OK]   {base}: {len(df):>6} barres | sha {sha[:12]} | {time.time()-t0:.1f}s")
    return base, sha


if __name__ == "__main__":
    print("=== GEL DES SNAPSHOTS NASDAQ (immuable, schema aligne sur l'or) ===\n")
    results = {}
    for tf in ("H1", "M15"):
        for side in ("BID", "ASK"):
            base, sha = freeze_one(tf, side)
            results[base] = sha
    print("\n=== RECAP ===")
    for base, sha in results.items():
        print(f"{base:52} {sha[:16] if sha else 'SKIP (deja gele)'}")
    print("\nDERNIERE_LIGNE_OK")
