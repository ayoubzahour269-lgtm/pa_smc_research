"""Familles fondees sur le temps : seance, plage d'ouverture, calendrier.

Ces trois pistes partagent une propriete inhabituelle : elles ne regardent presque pas
le prix. C'est precisement ce qui les rend interessantes a tester en premier — si un
effet purement horaire ou calendaire existe, toute famille plus sophistiquee qui inclut
la meme fenetre horaire heritera de son resultat sans rien apporter de plus.

Le temoin apparie a l'heure du protocole est ici particulierement severe, et c'est
voulu : il ne laissera passer ces familles que si elles apportent autre chose que le
choix du creneau.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from alphalab.alpha.base import AlphaFamily, Context
from alphalab.features import sessions


def _first_bar_of_window(index: pd.DatetimeIndex, window: sessions.SessionWindow) -> pd.Series:
    """Vrai sur la premiere barre de chaque occurrence contigue de la fenetre."""
    inside = window.mask(index).to_numpy()
    first = inside & ~np.concatenate([[False], inside[:-1]])
    return pd.Series(first, index=index)


class SessionDrift(AlphaFamily):
    """La derive d'une seance se prolonge-t-elle dans la suivante ?"""

    name = "seance_derive"
    question = (
        "Le sens du mouvement de la seance precedente, mesure une fois celle-ci close, "
        "predit-il celui de la seance suivante ?"
    )

    def __init__(self, source: str = "londres", trigger: str = "kz_newyork") -> None:
        self.source = source
        self.trigger = trigger

    def parameters(self) -> dict[str, Any]:
        return {"source": self.source, "trigger": self.trigger}

    def signal(self, ctx: Context) -> pd.Series:
        index = pd.DatetimeIndex(ctx.df.index)
        source = next(w for w in sessions.SESSIONS if w.name == self.source)
        all_windows = (*sessions.SESSIONS, *sessions.KILLZONES)
        trigger = next(w for w in all_windows if w.name == self.trigger)

        rng = sessions.completed_window_range(ctx.df, source)
        # Sens de la derniere occurrence close : position de la cloture courante par
        # rapport au milieu du range de cette seance. La comparaison n'utilise que des
        # valeurs deja connues.
        mid = rng[f"{self.source}_mid"]
        direction = (ctx.df["close"] - mid).apply(np.sign).fillna(0)

        fire = _first_bar_of_window(index, trigger)
        out: pd.Series = direction.where(fire, 0).astype(int).rename("signal")
        return out


class OpeningRangeBreakout(AlphaFamily):
    """La cassure de la plage d'ouverture d'une seance est-elle exploitable ?"""

    name = "opening_range"
    question = (
        "Apres la formation d'une plage d'ouverture, une cassure de cette plage "
        "annonce-t-elle une poursuite du mouvement ?"
    )

    def __init__(
        self, session: str = "kz_newyork", session_for_trade: str = "newyork"
    ) -> None:
        self.session = session
        self.session_for_trade = session_for_trade

    def parameters(self) -> dict[str, Any]:
        return {"plage": self.session, "fenetre_de_trade": self.session_for_trade}

    def signal(self, ctx: Context) -> pd.Series:
        index = pd.DatetimeIndex(ctx.df.index)
        all_windows = (*sessions.SESSIONS, *sessions.KILLZONES)
        opening = next(w for w in all_windows if w.name == self.session)
        trading = next(w for w in sessions.SESSIONS if w.name == self.session_for_trade)

        rng = sessions.completed_window_range(ctx.df, opening)
        high = rng[f"{self.session}_high"]
        low = rng[f"{self.session}_low"]

        close = ctx.df["close"]
        above = close > high
        below = close < low
        # Premiere cassure seulement : sans cela, chaque barre au-dessus de la plage
        # redeclencherait un signal et la densite exploserait.
        first_above = above & ~above.shift(1, fill_value=False)
        first_below = below & ~below.shift(1, fill_value=False)

        in_window = trading.mask(index)
        raw = np.where(first_above & in_window, 1, np.where(first_below & in_window, -1, 0))
        return pd.Series(raw, index=index, name="signal")


class TurnOfMonth(AlphaFamily):
    """L'effet de fin de mois documente sur les actions existe-t-il ici ?"""

    name = "fin_de_mois"
    question = (
        "Les derniers jours du mois et les premiers du suivant portent-ils un biais "
        "acheteur, comme documente sur les indices actions ?"
    )

    def __init__(
        self, days_before: int = 2, days_after: int = 3, trigger: str = "kz_newyork"
    ) -> None:
        self.days_before = days_before
        self.days_after = days_after
        self.trigger = trigger

    def parameters(self) -> dict[str, Any]:
        return {
            "jours_avant": self.days_before,
            "jours_apres": self.days_after,
            "declencheur": self.trigger,
        }

    def signal(self, ctx: Context) -> pd.Series:
        index = pd.DatetimeIndex(ctx.df.index)
        day_of_month = index.day
        days_in_month = index.days_in_month
        near_end = day_of_month > (days_in_month - self.days_before)
        near_start = day_of_month <= self.days_after
        window = pd.Series(near_end | near_start, index=index)

        all_windows = (*sessions.SESSIONS, *sessions.KILLZONES)
        trigger = next(w for w in all_windows if w.name == self.trigger)
        fire = _first_bar_of_window(index, trigger)
        return pd.Series(np.where(window & fire, 1, 0), index=index, name="signal")
