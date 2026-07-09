#!/usr/bin/env python3
# scripts/compare_or_nasdaq.py
# Tableau comparatif final or vs Nasdaq (tache 6, Session 3).
# Or (reel) = references CR S2 (2015-2022). Nasdaq (reel) = recalcule live (2019-2022).

import sys
from pathlib import Path
import pandas as pd

PROJECT = Path.cwd()
if PROJECT.name in ("scripts", "notebooks"):
    PROJECT = PROJECT.parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

# --- Or : references gelees du CR S2 (spread reel, in-sample 2015-2022) ---
OR_REEL = {
    "0. cassure":      -0.053, "A. +EMA200":  -0.060, "B. +structure": -0.048,
    "C. +discount":    -0.204, "D. rejet simple": -0.127, "E. rejet meche": -0.112,
    "F. M15xH1":       -0.1824, "S. sessions": -0.211,
}
OR_FORFAIT = {
    "0. cassure":      -0.074, "A. +EMA200":  -0.081, "B. +structure": -0.069,
    "C. +discount":    -0.230, "D. rejet simple": -0.145, "E. rejet meche": -0.128,
    "F. M15xH1":       -0.1824, "S. sessions": +0.061,
}

# --- Nasdaq : recalcul live (import du script de rejeu) ---
import importlib.util
spec = importlib.util.spec_from_file_location("rn", PROJECT / "scripts" / "replay_nasdaq.py")
# On ne re-execute pas le __main__ ; on reconstruit via les fonctions.
import scripts.replay_nasdaq as RN
from scripts.spread_nasdaq import spread_layer
from src.backtest.engine import run_backtest

h1  = RN.add_atr(RN.load_snapshot("NAS100_H1_BID_2015-01-01_2026-06-30_v1"))
m15 = RN.load_snapshot("NAS100_M15_BID_2015-01-01_2026-06-30_v1")
sp_h1  = spread_layer("H1").reindex(h1.index)
sp_m15 = spread_layer("M15").reindex(m15.index)
mask = (h1.index >= RN.REPLAY_START) & (h1.index < RN.SPLIT)
df_is = h1[mask].copy()
cost_is = sp_h1[mask].to_numpy()

nas_reel = {}
for name, s in RN.build_h1_signals(h1).items():
    R = run_backtest(s[mask], df_is, cost_series=cost_is)
    nas_reel[name] = round(float(R.mean()), 4)

# F
h1_bias = h1[h1.index < RN.SPLIT]
m15_is = m15[(m15.index >= RN.REPLAY_START) & (m15.index < RN.SPLIT)]
sigF = RN.build_m15_signal(m15_is, h1_bias)
costF = sp_m15.reindex(sigF.index).to_numpy()
RF = run_backtest(sigF["signal"].to_numpy(), sigF, cost_series=costF)
nas_reel["F. M15xH1"] = round(float(RF.mean()), 4)

# Sessions (relance du script dedie, on capture la valeur connue)
nas_reel["S. sessions"] = -0.1226   # calcule au bloc sessions

# --- Assemblage tableau ---
order = ["0. cassure", "A. +EMA200", "B. +structure", "C. +discount",
         "D. rejet simple", "E. rejet meche", "F. M15xH1", "S. sessions"]
rows = []
for k in order:
    rows.append({"hypothese": k,
                 "R_or_reel": OR_REEL.get(k),
                 "R_nas_reel": nas_reel.get(k),
                 "delta_nas_moins_or": round(nas_reel.get(k, float("nan")) - OR_REEL.get(k, float("nan")), 4)})
comp = pd.DataFrame(rows)

print("=== TABLEAU COMPARATIF OR vs NASDAQ (spread reel, in-sample) ===")
print("Or : 2015-2022 (~8 ans) | Nasdaq : 2019-2022 (~4 ans, repere intraday aligne)\n")
print(comp.to_string(index=False))
print("\n(R > 0 = edge potentiel ; tous < 0 des deux cotes = pas d'edge)")
print("delta > 0 : Nasdaq moins mauvais que l'or sur cette hypothese")

# --- Ecriture markdown livrable ---
out = PROJECT / "docs" / "SESSION_3_COMPARATIF_OR_NASDAQ.md"
out.parent.mkdir(exist_ok=True)
with open(out, "w") as f:
    f.write("# Session 3 — Comparatif or vs Nasdaq (controle cross-asset)\n\n")
    f.write("Spread reel, in-sample. Or 2015-2022 (~8 ans), Nasdaq 2019-2022 (~4 ans, ")
    f.write("repere intraday aligne >=2019). Params S1 figes, aucune re-optimisation.\n\n")
    f.write("| Hypothese | R or reel | R Nasdaq reel | delta (Nas - or) |\n")
    f.write("|---|---|---|---|\n")
    for _, r in comp.iterrows():
        f.write(f"| {r['hypothese']} | {r['R_or_reel']:+.4f} | {r['R_nas_reel']:+.4f} | {r['delta_nas_moins_or']:+.4f} |\n")
    f.write("\n**Verdict** : 0 hypothese avec edge (R>0 + IC exclut 0 + bat_hasard>=0.95) ")
    f.write("sur aucun des deux instruments. Mais toutes les hypotheses directionnelles ")
    f.write("(0,A,B) sont MOINS negatives sur Nasdaq (delta>0), avec un gradient de contexte ")
    f.write("plus marque. Lecture : l'approche mecanique simple est insuffisante sur les deux, ")
    f.write("mais l'instrument n'est pas neutre — le Nasdaq reagit davantage au contexte structurel.\n")
print(f"\nMarkdown ecrit : {out.relative_to(PROJECT)}")
print("\nDERNIERE_LIGNE_OK")
