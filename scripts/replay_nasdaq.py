#!/usr/bin/env python3
# scripts/replay_nasdaq.py — Rejeu des hypotheses S1 sur Nasdaq, au spread reel.
# CONTROLE cross-asset. Params S1 FIGES, aucune re-optimisation.
# Fenetre in-sample : 2019 <= t < 2023 (hold-out vierge).

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

from src.backtest.engine import run_backtest, evaluate
from scripts.spread_nasdaq import spread_layer

REPLAY_START = pd.Timestamp("2019-01-01", tz="UTC")
SPLIT        = pd.Timestamp("2023-01-01", tz="UTC")
N_BREAK, ATR_LEN = 20, 14


def load_snapshot(base):
    csv = SNAP / f"{base}.csv"
    man = json.loads((SNAP / f"{base}.manifest.json").read_text())
    assert hashlib.sha256(csv.read_bytes()).hexdigest() == man["sha256"], f"SHA256 diff {base}"
    d = pd.read_csv(csv, index_col=0, parse_dates=True)
    d.index.name = "timestamp"
    return d


def add_atr(df, atr_len=ATR_LEN):
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"],
                    (df["high"] - pc).abs(),
                    (df["low"] - pc).abs()], axis=1).max(axis=1)
    df["atr"] = tr.rolling(atr_len).mean()
    return df


def base_breakout_signal(df, n_break=N_BREAK):
    hh = df["high"].shift(1).rolling(n_break).max()
    ll = df["low"].shift(1).rolling(n_break).min()
    bull = df["close"] > hh
    bear = df["close"] < ll
    sig = pd.Series(0, index=df.index)
    sig[bull & ~bull.shift(1, fill_value=False)] = 1
    sig[bear & ~bear.shift(1, fill_value=False)] = -1
    return sig.to_numpy()


def daily_bias_table(h1_df, W=3):
    daily = (h1_df.resample("1D")
             .agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna())
    Hd, Ld, Cd = daily["high"].values, daily["low"].values, daily["close"].values
    nD = len(daily)
    is_ph = np.zeros(nD, bool); is_pl = np.zeros(nD, bool)
    for k in range(W, nD - W):
        if Hd[k] == Hd[k - W:k + W + 1].max(): is_ph[k] = True
        if Ld[k] == Ld[k - W:k + W + 1].min(): is_pl[k] = True
    bias = np.zeros(nD, int); last_sh, last_sl, cur = np.nan, np.nan, 0
    sh_arr = np.full(nD, np.nan); sl_arr = np.full(nD, np.nan)
    for t in range(nD):
        k = t - W
        if k >= 0:
            if is_ph[k]: last_sh = Hd[k]
            if is_pl[k]: last_sl = Ld[k]
        if not np.isnan(last_sh) and Cd[t] > last_sh: cur = 1
        elif not np.isnan(last_sl) and Cd[t] < last_sl: cur = -1
        bias[t] = cur
        sh_arr[t], sl_arr[t] = last_sh, last_sl
    return pd.DataFrame({"known_from": daily.index + pd.Timedelta(days=1),
                         "bias": bias, "sh": sh_arr, "sl": sl_arr})


def build_h1_signals(h1_df):
    df = h1_df.copy()
    base = base_breakout_signal(df)
    ema200 = df["close"].ewm(span=200, adjust=False).mean()
    htf_ema = np.where(df["close"] > ema200, 1, -1)
    tbl = daily_bias_table(df)
    m = pd.merge_asof(df.reset_index()[["timestamp"]], tbl,
                      left_on="timestamp", right_on="known_from", direction="backward")
    strc = m["bias"].fillna(0).astype(int).values
    sh, sl = m["sh"].values, m["sl"].values
    eq = (sh + sl) / 2.0
    o, h, l, c = (df[x].values for x in ["open", "high", "low", "close"])
    sig_A = np.where(base == htf_ema, base, 0)
    sig_B = np.where(base == strc, base, 0)
    discount = c < eq; premium = c > eq
    in_zone = np.where(strc == 1, discount, np.where(strc == -1, premium, False)).astype(bool)
    sig_C = np.where((sig_B != 0) & in_zone, sig_B, 0)
    body = np.abs(c - o); lower_wick = np.minimum(o, c) - l; upper_wick = h - np.maximum(o, c)
    bull_simple = (strc == 1) & discount & (c > o)
    bear_simple = (strc == -1) & premium & (c < o)
    sig_D = np.where(bull_simple, 1, np.where(bear_simple, -1, 0))
    bull_wick = bull_simple & (lower_wick >= body)
    bear_wick = bear_simple & (upper_wick >= body)
    sig_E = np.where(bull_wick, 1, np.where(bear_wick, -1, 0))
    return {"0. cassure": base, "A. +EMA200": sig_A, "B. +structure": sig_B,
            "C. +discount": sig_C, "D. rejet simple": sig_D, "E. rejet meche": sig_E}


def build_m15_signal(m15_df, h1_df, atr_len=ATR_LEN, n_break=N_BREAK):
    df = m15_df.copy()
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(),
                    (df["low"] - pc).abs()], axis=1).max(axis=1)
    df["atr"] = tr.rolling(atr_len).mean()
    tbl = daily_bias_table(h1_df)
    m = pd.merge_asof(df.reset_index()[["timestamp"]], tbl[["known_from", "bias"]],
                      left_on="timestamp", right_on="known_from", direction="backward")
    df["bias"] = m["bias"].fillna(0).astype(int).values
    up   = (df["close"] > df["open"]) & (df["close"].shift(1) < df["open"].shift(1))
    down = (df["close"] < df["open"]) & (df["close"].shift(1) > df["open"].shift(1))
    df["signal"] = np.where((df["bias"] == 1) & up.values, 1,
                    np.where((df["bias"] == -1) & down.values, -1, 0))
    return df.dropna(subset=["atr"])


def judge_R(R, seed=0, n_boot=10000):
    n = len(R)
    if n == 0:
        return {"trades": 0, "R": np.nan, "IC": [np.nan, np.nan], "exclut_0": False}
    rng = np.random.default_rng(seed)
    boot = np.array([rng.choice(R, n, replace=True).mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"trades": n, "R": round(float(R.mean()), 4),
            "IC": [round(float(lo), 4), round(float(hi), 4)],
            "exclut_0": bool(lo > 0 or hi < 0)}


def bat_hasard_reel(df, cost_series, R_reel_mean, N, density, p_long,
                    n_random=100, seed0=1000):
    def one(s):
        r = np.random.default_rng(s)
        rand_sig = np.where(r.random(N) < density,
                            np.where(r.random(N) < p_long, 1, -1), 0)
        return run_backtest(rand_sig, df, cost_series=cost_series).mean()
    rand = np.array([one(seed0 + s) for s in range(n_random)])
    return round(float((R_reel_mean > rand).mean()), 2)


if __name__ == "__main__":
    pd.set_option("display.width", 240, "display.max_colwidth", 32)
    print("=== REJEU NASDAQ — CONTROLE CROSS-ASSET ===")
    print(f"Fenetre in-sample : {REPLAY_START.date()} <= t < {SPLIT.date()} (hold-out vierge)")
    print(f"Params FIGES : N_BREAK={N_BREAK}, ATR_LEN={ATR_LEN}, W=3 (aucune re-optimisation)\n")

    h1  = add_atr(load_snapshot("NAS100_H1_BID_2015-01-01_2026-06-30_v1"))
    m15 = load_snapshot("NAS100_M15_BID_2015-01-01_2026-06-30_v1")
    sp_h1  = spread_layer("H1").reindex(h1.index)
    sp_m15 = spread_layer("M15").reindex(m15.index)

    mask_is_h1 = (h1.index >= REPLAY_START) & (h1.index < SPLIT)
    df_is = h1[mask_is_h1].copy()
    cost_h1_is = sp_h1[mask_is_h1].to_numpy()

    signals = build_h1_signals(h1)
    rows = []
    for name, s in signals.items():
        s_is = s[mask_is_h1]
        R_forf = run_backtest(s_is, df_is)
        R_reel = run_backtest(s_is, df_is, cost_series=cost_h1_is)
        jf, jr = judge_R(R_forf), judge_R(R_reel)
        mask = s_is != 0
        density = mask.mean()
        p_long = (s_is == 1).sum() / max(mask.sum(), 1)
        bh = bat_hasard_reel(df_is, cost_h1_is, R_reel.mean(),
                             len(df_is), density, p_long)
        rows.append({"hypothese": name, "trades": jr["trades"],
                     "R_forfait": jf["R"], "R_reel": jr["R"],
                     "IC_reel": jr["IC"], "exclut_0": jr["exclut_0"], "bat_hasard": bh})

    h1_bias_src = h1[h1.index < SPLIT]
    m15_is = m15[(m15.index >= REPLAY_START) & (m15.index < SPLIT)]
    sigF = build_m15_signal(m15_is, h1_bias_src)
    dfF = sigF.copy()
    sF = dfF["signal"].to_numpy()
    costF = sp_m15.reindex(dfF.index).to_numpy()
    nanF = int(np.isnan(costF).sum())
    RF_forf = run_backtest(sF, dfF)
    RF_reel = run_backtest(sF, dfF, cost_series=costF)
    jfF, jrF = judge_R(RF_forf), judge_R(RF_reel)
    maskF = sF != 0
    bhF = bat_hasard_reel(dfF, costF, RF_reel.mean(), len(dfF),
                          maskF.mean(), (sF == 1).sum() / max(maskF.sum(), 1))
    rows.append({"hypothese": "F. M15xH1", "trades": jrF["trades"],
                 "R_forfait": jfF["R"], "R_reel": jrF["R"],
                 "IC_reel": jrF["IC"], "exclut_0": jrF["exclut_0"], "bat_hasard": bhF})

    comp = pd.DataFrame(rows)
    print("=== RESULTATS NASDAQ (in-sample 2019-2022, spread reel) ===")
    print(f"(F: NaN cout M15 = {nanF}, repli forfait 0.40 sur ces barres)\n")
    print(comp.to_string(index=False))
    print("\nJuge : 'interessant' SSI IC_reel exclut 0 ET bat_hasard >= 0.95 ET R_reel > 0")
    inter = comp[(comp["exclut_0"]) & (comp["bat_hasard"] >= 0.95) & (comp["R_reel"] > 0)]
    print(f"\nHypotheses interessantes : {len(inter)}")
    if len(inter):
        print(inter.to_string(index=False))
    else:
        print(">>> AUCUN signe de vie cote Nasdaq (memes regles, spread reel).")
    print("\nDERNIERE_LIGNE_OK")
