"""Moteur de backtest — Projet A (price action, XAU/USD).
Extrait du notebook 01 à l'étape de mise au propre. Ne pas modifier sans relancer les tests.
"""
import numpy as np
import pandas as pd

SL_MULT, TP_MULT, MAX_HOLD, COST_USD = 1.5, 3.0, 100, 0.40


def run_backtest(signal, df, sl_mult=SL_MULT, tp_mult=TP_MULT,
                 max_hold=MAX_HOLD, cost_usd=COST_USD, cost_series=None):
    """Un trade à la fois, entrée à l'ouverture de la barre suivante,
    SL/TP en multiples d'ATR. Renvoie le tableau des R réalisés."""
    o, h, l, c = (df[x].to_numpy() for x in ["open", "high", "low", "close"])
    atr = df["atr"].to_numpy()
    sig = np.asarray(signal)
    n = len(df)
    cs = None if cost_series is None else np.asarray(cost_series, dtype=float)
    if cs is not None and len(cs) != n:
        raise ValueError("cost_series length must equal df length")
    R_list, i = [], 0
    while i < n - 1:
        s = sig[i]
        if s != 0 and not np.isnan(atr[i]):
            d = 1 if s == 1 else -1
            entry = o[i + 1]
            R_usd = sl_mult * atr[i]
            if cs is None:
                cost = cost_usd
            else:
                cost = cs[i + 1]
                if np.isnan(cost):
                    cost = cost_usd
            sl = entry - d * sl_mult * atr[i]
            tp = entry + d * tp_mult * atr[i]
            ep, ej = None, None
            end = min(i + 1 + max_hold, n)
            for j in range(i + 1, end):
                hit_sl = (l[j] <= sl) if d == 1 else (h[j] >= sl)
                hit_tp = (h[j] >= tp) if d == 1 else (l[j] <= tp)
                if hit_sl:
                    ep, ej = sl, j
                    break
                if hit_tp:
                    ep, ej = tp, j
                    break
            if ep is None:
                ej = end - 1
                ep = c[ej]
            R_list.append((d * (ep - entry) - cost) / R_usd)
            i = ej + 1
        else:
            i += 1
    return np.array(R_list)


def evaluate(signal, df, label, n_boot=10000, n_random=100, seed=0):
    """Espérance-R + IC bootstrap + témoin aléatoire apparié (densité ET biais long/court)."""
    signal = np.asarray(signal)
    R = run_backtest(signal, df)
    n = len(R)
    rng = np.random.default_rng(seed)
    boot = np.array([rng.choice(R, n, replace=True).mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    mask = signal != 0
    density = mask.mean()
    p_long = (signal == 1).sum() / max(mask.sum(), 1)
    N = len(df)

    def one_random(s):
        r = np.random.default_rng(s)
        rand_sig = np.where(r.random(N) < density,
                            np.where(r.random(N) < p_long, 1, -1), 0)
        return run_backtest(rand_sig, df).mean()

    rand = np.array([one_random(1000 + s) for s in range(n_random)])
    return {"label": label, "trades": n, "long_frac": round(float(p_long), 2),
            "esperance_R": round(float(R.mean()), 4),
            "IC": [round(float(lo), 4), round(float(hi), 4)],
            "exclut_0": bool(lo > 0 or hi < 0),
            "hasard": round(float(rand.mean()), 4),
            "bat_hasard": round(float((R.mean() > rand).mean()), 2)}