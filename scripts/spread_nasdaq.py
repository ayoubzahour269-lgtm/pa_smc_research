#!/usr/bin/env python3
# scripts/spread_nasdaq.py
# Couche de spread reel Nasdaq : ASK.open - BID.open, barre a barre.
# Meme methode que l'or (S2). Importable par le notebook de rejeu.

import json, hashlib
import pandas as pd
from pathlib import Path

PROJECT = Path.cwd()
if PROJECT.name in ("scripts", "notebooks"):
    PROJECT = PROJECT.parent
SNAP = PROJECT / "data" / "snapshots"


def load_snapshot(base):
    csv = SNAP / f"{base}.csv"
    man = json.loads((SNAP / f"{base}.manifest.json").read_text())
    sha = hashlib.sha256(csv.read_bytes()).hexdigest()
    assert sha == man["sha256"], f"SHA256 different pour {base} !"
    d = pd.read_csv(csv, index_col=0, parse_dates=True)
    d.index.name = "timestamp"
    return d


def spread_layer(tf):
    """Retourne la Series de spread (ASK.open - BID.open) pour tf in {H1, M15}."""
    bid = load_snapshot(f"NAS100_{tf}_BID_2015-01-01_2026-06-30_v1")
    ask = load_snapshot(f"NAS100_{tf}_ASK_2015-01-01_2026-06-30_v1")
    if not bid.index.equals(ask.index):
        common = bid.index.intersection(ask.index)
        bid, ask = bid.loc[common], ask.loc[common]
    return (ask["open"] - bid["open"]).rename("spread")


if __name__ == "__main__":
    for tf in ("H1", "M15"):
        sp = spread_layer(tf)
        neg = int((sp < 0).sum())
        print(f"=== SPREAD NASDAQ {tf} (ASK.open - BID.open) ===")
        print(f"  barres={len(sp)} | negatifs={neg} | "
              f"mediane={sp.median():.3f} | moyenne={sp.mean():.3f} | "
              f"p95={sp.quantile(0.95):.3f} pts")
    print("\nDERNIERE_LIGNE_OK")
