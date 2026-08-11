"""Familles de retour a la moyenne : ecart au VWAP, gap d'ouverture.

Ces pistes parient sur l'inverse des familles de tendance. Les tester cote a cote sur
le meme protocole est le seul moyen de savoir laquelle des deux intuitions — "ca
continue" ou "ca revient" — decrit ces instruments a cet horizon. Il est parfaitement
possible qu'aucune des deux ne le fasse.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from alphalab.alpha.base import AlphaFamily, Context
from alphalab.features import indicators, sessions


class VwapReversion(AlphaFamily):
    """Un ecart marque au VWAP du jour se resorbe-t-il ?"""

    name = "retour_vwap"
    question = (
        "Lorsque le prix s'ecarte fortement du VWAP ancre sur la journee de trading, "
        "revient-il vers lui plus souvent qu'il ne poursuit ?"
    )

    def __init__(
        self, z_threshold: float = 2.0, window: int = 48, session: str = "newyork"
    ) -> None:
        self.z_threshold = z_threshold
        self.window = window
        self.session = session

    def parameters(self) -> dict[str, Any]:
        return {"seuil_z": self.z_threshold, "fenetre": self.window, "seance": self.session}

    def signal(self, ctx: Context) -> pd.Series:
        index = pd.DatetimeIndex(ctx.df.index)
        anchor = sessions.trading_day(index)
        vwap = indicators.anchored_vwap(ctx.df, anchor)
        deviation = ctx.df["close"] - vwap
        z = indicators.zscore(deviation, self.window)

        in_session = next(w for w in sessions.SESSIONS if w.name == self.session).mask(index)
        # On fade l'ecart : trop haut -> vente, trop bas -> achat.
        raw = np.where(
            (z >= self.z_threshold) & in_session,
            -1,
            np.where((z <= -self.z_threshold) & in_session, 1, 0),
        )
        signal = pd.Series(raw, index=index, name="signal")
        return signal.where(signal != signal.shift(1, fill_value=0), 0)


class OvernightGap(AlphaFamily):
    """Un gap d'ouverture se comble-t-il ?"""

    name = "gap_ouverture"
    question = (
        "Lorsque la premiere barre d'une journee de trading ouvre loin de la cloture "
        "precedente, le prix revient-il combler cet ecart ?"
    )

    def __init__(self, min_atr: float = 0.5, fade: bool = True) -> None:
        self.min_atr = min_atr
        self.fade = fade

    def parameters(self) -> dict[str, Any]:
        return {"gap_min_atr": self.min_atr, "sens": "comblement" if self.fade else "poursuite"}

    def signal(self, ctx: Context) -> pd.Series:
        index = pd.DatetimeIndex(ctx.df.index)
        day = sessions.trading_day(index)
        is_first = day != day.shift(1)

        gap = ctx.df["open"] - ctx.df["close"].shift(1)
        size_atr = gap / ctx.features["atr"]

        big_up = is_first & (size_atr >= self.min_atr)
        big_down = is_first & (size_atr <= -self.min_atr)

        # Comblement : un gap haussier se vend. Poursuite : il s'achete.
        up_direction = -1 if self.fade else 1
        raw = np.where(big_up, up_direction, np.where(big_down, -up_direction, 0))
        return pd.Series(raw, index=index, name="signal")


class RangeFade(AlphaFamily):
    """Les extremes de la seance asiatique tiennent-ils pendant la seance de Londres ?"""

    name = "fade_range_asiatique"
    question = (
        "Le premier contact avec un extreme du range asiatique, une fois cette seance "
        "close, provoque-t-il un rejet exploitable pendant Londres ?"
    )

    def __init__(self, source: str = "asie", trade_session: str = "londres") -> None:
        self.source = source
        self.trade_session = trade_session

    def parameters(self) -> dict[str, Any]:
        return {"range_source": self.source, "seance_de_trade": self.trade_session}

    def signal(self, ctx: Context) -> pd.Series:
        index = pd.DatetimeIndex(ctx.df.index)
        source = next(w for w in sessions.SESSIONS if w.name == self.source)
        rng = sessions.completed_window_range(ctx.df, source)
        high = rng[f"{self.source}_high"]
        low = rng[f"{self.source}_low"]

        touch_high = (ctx.df["high"] >= high) & (ctx.df["close"] < high)
        touch_low = (ctx.df["low"] <= low) & (ctx.df["close"] > low)

        window = next(w for w in sessions.SESSIONS if w.name == self.trade_session).mask(index)
        raw = np.where(touch_high & window, -1, np.where(touch_low & window, 1, 0))
        signal = pd.Series(raw, index=index, name="signal")
        return signal.where(signal != signal.shift(1, fill_value=0), 0)
