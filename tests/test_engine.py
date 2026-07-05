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