"""Contrat de donnees OHLCV et sa validation stricte.

Une seule fonction fait autorite : `validate_ohlcv`. Tout ce qui entre dans la
plateforme passe par elle. Les erreurs sont volontairement bavardes — un contrat
viole silencieusement se paye plus tard en resultat de backtest faux.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd

OHLC_COLUMNS: Final[tuple[str, ...]] = ("open", "high", "low", "close")
OHLCV_COLUMNS: Final[tuple[str, ...]] = (*OHLC_COLUMNS, "volume")


class SchemaError(ValueError):
    """Une trame ne respecte pas le contrat OHLCV."""


def validate_ohlcv(df: pd.DataFrame, *, name: str = "<frame>", require_volume: bool = True) -> None:
    """Verifie le contrat OHLCV. Leve `SchemaError` au premier manquement.

    Contrat :
      - index `DatetimeIndex` tz-aware en UTC, strictement croissant, sans doublon ;
      - colonnes open/high/low/close (+ volume si demande), numeriques, sans NaN ;
      - coherence des barres : low <= min(open, close) <= max(open, close) <= high.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise SchemaError(
            f"{name}: l'index doit etre un DatetimeIndex, recu {type(df.index).__name__}"
        )
    if df.index.tz is None:
        raise SchemaError(f"{name}: l'index doit etre tz-aware (UTC), il est naif")
    if str(df.index.tz) != "UTC":
        raise SchemaError(f"{name}: l'index doit etre en UTC, recu {df.index.tz}")
    if len(df) == 0:
        raise SchemaError(f"{name}: trame vide")
    if not df.index.is_monotonic_increasing:
        raise SchemaError(f"{name}: l'index doit etre croissant")
    n_dup = int(df.index.duplicated().sum())
    if n_dup:
        raise SchemaError(f"{name}: {n_dup} horodatage(s) en doublon")

    required = OHLCV_COLUMNS if require_volume else OHLC_COLUMNS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SchemaError(f"{name}: colonnes manquantes {missing}")

    for col in required:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise SchemaError(f"{name}: la colonne '{col}' n'est pas numerique ({df[col].dtype})")

    for col in OHLC_COLUMNS:
        n_nan = int(df[col].isna().sum())
        if n_nan:
            raise SchemaError(f"{name}: {n_nan} NaN dans la colonne '{col}'")

    o, h, low, c = (df[x].to_numpy(dtype=float) for x in OHLC_COLUMNS)
    bad_high = int(np.sum(h < np.maximum(o, c)))
    bad_low = int(np.sum(low > np.minimum(o, c)))
    bad_range = int(np.sum(h < low))
    if bad_high or bad_low or bad_range:
        raise SchemaError(
            f"{name}: incoherences OHLC — high<max(o,c): {bad_high}, "
            f"low>min(o,c): {bad_low}, high<low: {bad_range}"
        )


def bar_grid_report(df: pd.DataFrame, expected: pd.Timedelta) -> dict[str, object]:
    """Statistiques de regularite de la grille temporelle.

    Ne leve rien : les marches ont des week-ends et des jours feries, les trous sont
    normaux. Sert au diagnostic (detection de changement de regime de seance) et aux
    manifestes, pas a la validation.
    """
    deltas = df.index.to_series().diff().dropna()
    if deltas.empty:
        return {"n_bars": int(len(df)), "n_gaps": 0, "largest_gap": None, "median_delta": None}
    gaps = deltas[deltas > expected]
    return {
        "n_bars": int(len(df)),
        "n_gaps": int(len(gaps)),
        "largest_gap": str(deltas.max()),
        "median_delta": str(deltas.median()),
        "bars_per_day": round(float(len(df)) / max((df.index[-1] - df.index[0]).days, 1), 3),
    }
