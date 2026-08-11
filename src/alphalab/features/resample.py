"""Agregation multi-timeframe sans anticipation.

Deuxieme piege classique du multi-timeframe, apres celui des pivots : utiliser la barre
H4 *en cours* pour decider sur une barre M15. La barre H4 qui couvre 12h-16h n'est
connue qu'a 16h ; s'en servir a 12h15 revient a connaitre les quatre heures suivantes.

`align_completed` resout ce point une fois pour toutes : elle ne rend jamais qu'une
barre superieure DEJA CLOSE.
"""

from __future__ import annotations

from collections.abc import Hashable
from typing import Final

import pandas as pd

from alphalab.config import BAR_DURATION
from alphalab.data.schema import validate_ohlcv

#: Regles pandas correspondant aux timeframes du projet.
RESAMPLE_RULE: Final[dict[str, str]] = {
    "M1": "1min",
    "M5": "5min",
    "M15": "15min",
    "H1": "1h",
    "H4": "4h",
    "D1": "1D",
}

_AGG: Final[dict[str, str]] = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum",
}


def to_timeframe(df: pd.DataFrame, timeframe: str, *, origin_hour: int = 0) -> pd.DataFrame:
    """Agrege vers un timeframe superieur.

    L'horodatage rendu est celui de l'OUVERTURE de la barre agregee : la barre H4
    etiquetee 12:00 couvre [12:00, 16:00). C'est la convention du reste du projet, et
    c'est aussi ce qui rend `align_completed` lisible.

    `origin_hour` decale la grille (utile pour aligner les barres journalieres sur le
    rollover a 22h plutot que sur minuit UTC).
    """
    if timeframe not in RESAMPLE_RULE:
        raise ValueError(f"Timeframe inconnu : {timeframe}. Connus : {sorted(RESAMPLE_RULE)}")
    columns: dict[Hashable, str] = {k: v for k, v in _AGG.items() if k in df.columns}
    origin = pd.DatetimeIndex(df.index)[0].normalize() + pd.Timedelta(hours=origin_hour)
    resampled = df.resample(
        RESAMPLE_RULE[timeframe], label="left", closed="left", origin=origin
    )
    out = resampled.agg(columns)
    out = out.dropna(subset=["open", "high", "low", "close"])
    out.index.name = "timestamp"
    validate_ohlcv(out, name=f"resample->{timeframe}", require_volume="volume" in columns)
    return out


def align_completed(
    base_index: pd.DatetimeIndex,
    htf: pd.DataFrame,
    timeframe: str,
    *,
    prefix: str | None = None,
) -> pd.DataFrame:
    """Aligne des barres superieures CLOSES sur un index plus fin.

    Pour chaque horodatage `t` de `base_index`, rend la derniere barre de `htf` dont la
    cloture (ouverture + duree) est <= `t`. La barre en cours est donc exclue par
    construction, quelle que soit l'inattention de l'appelant.
    """
    if timeframe not in BAR_DURATION:
        raise ValueError(f"Timeframe inconnu : {timeframe}")
    tag = prefix if prefix is not None else f"{timeframe.lower()}_"

    available_at = pd.DatetimeIndex(htf.index) + BAR_DURATION[timeframe]
    known = htf.copy()
    known.index = available_at
    known = known[~known.index.duplicated(keep="last")].sort_index()

    aligned = known.reindex(known.index.union(base_index)).ffill().reindex(base_index)
    aligned.columns = pd.Index([f"{tag}{c}" for c in aligned.columns])
    aligned.index = base_index
    return aligned


def multi_timeframe_context(
    df: pd.DataFrame, timeframes: tuple[str, ...] = ("H1", "H4", "D1")
) -> pd.DataFrame:
    """Contexte superieur complet, aligne et strictement clos, pour une trame de base."""
    base_index = pd.DatetimeIndex(df.index)
    parts = [align_completed(base_index, to_timeframe(df, tf), tf) for tf in timeframes]
    return pd.concat(parts, axis=1)
