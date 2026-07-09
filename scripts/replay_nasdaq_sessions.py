#!/usr/bin/env python3
# scripts/replay_nasdaq_sessions.py
# Hypothese "sessions 21h->23h" rejouee sur Nasdaq, cout reel spread(21h)+spread(23h).
# Juge maison (sess_R + bootstrap), identique S1. Params FIGES. In-sample 2019-2022.

import sys, json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT = Path.cwd()
if PROJECT.name in ("scripts", "notebooks"):
    PROJECT = PROJECT.parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
SNAP = PROJECT / "data" / "snapshots"
from scripts.spread_nasdaq import spread_layer

REPLAY_START = pd.Timestamp("2019-01-01", tz="UTC")
SPLIT        = pd.Timestamp("2023-01-01", tz="UTC")
ENTRY_H, EXIT_H = 21, 23


def load_snapshot(base):
    csv = SNAP / f"{base}.csv"
    man = json.loads((SNAP / f"{base}.manifest.json").read_text())
    assert hashlib.sha256(csv.read_bytes()).hexdigest() == man["sha256"], f"SHA256 diff {base}"
    d = pd.read_csv(csv, index_col=0, parse_dates=True)
    d.index.name = "timestamp"
    return d


def judge(R, label, seed=0, n_boot=5000):
    n = len(R)
    if n == 0:
        print(f"{label:34s} : aucun trade"); return
    rng = np.random.default_rng(seed)
    boot = np.array([rng.choice(R, n, replace=True).mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    excl = "OUI" if (lo > 0 or hi < 0) else "non"
    print(f"{label:34s} n={n:4d}  esp={R.mean():+.4f} R  IC=[{lo:+.4f},{hi:+.4f}]  exclut0={excl}")


if __name__ == "__main__":
    print("=== NASDAQ — hypothese sessions 21h->23h (in-sample 2019-2022) ===\n")

    bid = load_snapshot("NAS100_H1_BID_2015-01-01_2026-06-30_v1")
    w = bid[(bid.index >= REPLAY_START) & (bid.index < SPLIT)].copy()
    w["hour"] = w.index.hour
    w["date"] = w.index.normalize()

    piv = w.pivot_table(index="date", columns="hour", values="close", aggfunc="last")
    day_scale = w.groupby("date")["close"].std()

    # Cout reel par jour : spread(21h) + spread(23h)
    sp = spread_layer("H1")
    sp = sp[(sp.index >= REPLAY_START) & (sp.index < SPLIT)]
    sp_df = pd.DataFrame({"spread": sp.values}, index=sp.index)
    sp_df["hour"] = sp_df.index.hour
    sp_df["date"] = sp_df.index.normalize()
    sp21 = sp_df[sp_df["hour"] == ENTRY_H].set_index("date")["spread"]
    sp23 = sp_df[sp_df["hour"] == EXIT_H].set_index("date")["spread"]
    cost_reel_jour = sp21.add(sp23, fill_value=np.nan).dropna()

    print("Cout reel session (spread 21h + 23h) :")
    print(f"  mediane={cost_reel_jour.median():.3f} | moyenne={cost_reel_jour.mean():.3f} | "
          f"p95={cost_reel_jour.quantile(0.95):.3f} pts")
    print(f"  (rappel profil : 21h liquide ~1.3 pts, 22h illiquide ~3.4 pts)\n")

    def sess_R(cost_scalar_or_series):
        cols = {"e": piv.get(ENTRY_H), "x": piv.get(EXIT_H), "s": day_scale}
        if isinstance(cost_scalar_or_series, pd.Series):
            cols["c"] = cost_scalar_or_series
        d = pd.DataFrame(cols).dropna()
        d = d[d["s"] > 0]
        c = d["c"] if isinstance(cost_scalar_or_series, pd.Series) else cost_scalar_or_series
        return (((d["x"] - d["e"]) - c) / d["s"]).values

    print("Strategie : LONG close 21h -> close 23h UTC\n")
    print(f"Rendement brut moyen/trade : {sess_R(0.0).mean():+.4f} R (avant cout)\n")
    judge(sess_R(0.0),             "Brut (cout 0)")
    judge(sess_R(cost_reel_jour),  "Cout REEL (spread 21h+23h)")
    print("\nJuge : interessant SSI IC exclut 0 du bon cote (esp > 0).")
    print("\nDERNIERE_LIGNE_OK")
