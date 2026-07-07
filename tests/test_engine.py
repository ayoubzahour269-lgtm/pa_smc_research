"""Test de non-régression du moteur de backtest.
Vérifie que run_backtest reproduit le résultat de référence de l'hypothèse #1
(cassure 20 barres) : ~2662 trades, espérance-R ~ -0.0524.
Si ce test échoue, c'est que le moteur a changé de comportement.
"""
import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.backtest.engine import run_backtest


def _load_snapshot():
    snap_dir = ROOT / "data" / "snapshots"
    base = "XAUUSD_H1_BID_2015-01-01_2026-06-30_v1"
    csv_path = snap_dir / f"{base}.csv"
    manifest = json.loads((snap_dir / f"{base}.manifest.json").read_text())
    sha_now = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    assert sha_now == manifest["sha256"], "SHA256 du snapshot ne correspond pas au manifeste"
    data = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    return data


def _build_signal(data, n_break=20, atr_len=14):
    df = data.copy()
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["atr"] = tr.rolling(atr_len).mean()
    hh = df["high"].shift(1).rolling(n_break).max()
    ll = df["low"].shift(1).rolling(n_break).min()
    bull = df["close"] > hh
    bear = df["close"] < ll
    df["signal"] = 0
    df.loc[bull & ~bull.shift(1, fill_value=False), "signal"] = 1
    df.loc[bear & ~bear.shift(1, fill_value=False), "signal"] = -1
    df = df.dropna(subset=["atr"])
    return df


def test_snapshot_integrity():
    """Le snapshot doit exister et son sha256 doit matcher le manifeste."""
    data = _load_snapshot()
    assert len(data) == 68041


def test_hypothese1_reproductible():
    """Le moteur doit redonner ~2662 trades et une espérance-R ~ -0.0524."""
    data = _load_snapshot()
    df = _build_signal(data)
    R = run_backtest(df["signal"].values, df)
    assert 2600 <= len(R) <= 2720, f"Nombre de trades inattendu : {len(R)}"
    esp = float(R.mean())
    assert -0.060 <= esp <= -0.045, f"Esperance-R hors reference : {esp:.4f}"


def _tiny_time_exit_df():
    """5 barres, ATR=1, un long en barre 0 (entree a open[1]).
    Prix plats -> ni SL ni TP -> sortie time-exit au close final.
    Avec close_final == entry, (ep - entry) ~ 0 => le R isole le cout."""
    idx = pd.date_range("2020-01-01", periods=5, freq="h", tz="UTC")
    price = 2000.0
    df = pd.DataFrame(
        {"open": price, "high": price + 0.2, "low": price - 0.2,
         "close": price, "atr": 1.0},
        index=idx,
    )
    signal = np.array([1, 0, 0, 0, 0])
    return df, signal


def test_cost_series_constante_equivaut_au_forfait():
    """cost_series constant = k partout doit reproduire exactement cost_usd=k."""
    df, signal = _tiny_time_exit_df()
    k = 0.40
    R_forfait = run_backtest(signal, df, cost_usd=k)
    R_series = run_backtest(signal, df, cost_series=np.full(len(df), k))
    assert len(R_forfait) == len(R_series) == 1
    assert np.allclose(R_forfait, R_series), (R_forfait, R_series)


def test_cost_series_plus_cher_reduit_le_R():
    """Un spread reel plus cher que le forfait doit faire baisser le R."""
    df, signal = _tiny_time_exit_df()
    R_cheap = run_backtest(signal, df, cost_series=np.full(len(df), 0.40))
    R_expensive = run_backtest(signal, df, cost_series=np.full(len(df), 2.0))
    assert R_expensive[0] < R_cheap[0], (R_cheap, R_expensive)


def test_cost_series_mauvaise_longueur_leve_valueerror():
    """Un cost_series de longueur != n doit lever ValueError (garde-fou)."""
    df, signal = _tiny_time_exit_df()
    import pytest
    with pytest.raises(ValueError):
        run_backtest(signal, df, cost_series=np.full(len(df) - 1, 0.40))
