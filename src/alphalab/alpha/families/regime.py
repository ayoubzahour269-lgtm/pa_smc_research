"""Mecanismes absents de la premiere campagne.

La mesure de diversite a montre que le catalogue initial explorait moins large que son
compte ne le suggerait. Ces familles comblent les angles morts identifies : elles
n'utilisent ni le declencheur de cassure, ni les extremes de seance, ni les pivots de
structure.

Trois idees vraiment differentes :

  - le regime de volatilite comme **etat** qui change le sens du pari, et non comme
    simple filtre qui l'autorise ou l'interdit ;
  - un **horizon multi-jours**, alors que tout le reste du catalogue est intraday ;
  - une **saisonnalite intraday** estimee sur l'historique passe, cellule par cellule
    (heure x jour de semaine), sans jamais utiliser le futur.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from alphalab.alpha.base import AlphaFamily, Context
from alphalab.features import indicators, sessions


class VolatilityRegimeSwitch(AlphaFamily):
    """Le marche change-t-il de nature selon son regime de volatilite ?"""

    name = "regime_volatilite"
    question = (
        "Le prix suit-il sa tendance en volatilite elevee et revient-il vers sa moyenne "
        "en volatilite faible — autrement dit, le regime change-t-il le SENS du pari et "
        "pas seulement son opportunite ?"
    )

    def __init__(
        self, window: int = 240, high_q: float = 0.7, low_q: float = 0.3, z_len: int = 24
    ) -> None:
        self.window = window
        self.high_q = high_q
        self.low_q = low_q
        self.z_len = z_len

    def parameters(self) -> dict[str, Any]:
        return {
            "fenetre_regime": self.window,
            "quantile_haut": self.high_q,
            "quantile_bas": self.low_q,
            "longueur_z": self.z_len,
        }

    def signal(self, ctx: Context) -> pd.Series:
        index = pd.DatetimeIndex(ctx.df.index)
        volatility = ctx.features["atr"]
        # Quantiles glissants et decales : le seuil ne doit pas dependre de la barre
        # qu'il qualifie.
        high_threshold = volatility.shift(1).rolling(self.window).quantile(self.high_q)
        low_threshold = volatility.shift(1).rolling(self.window).quantile(self.low_q)

        z = indicators.zscore(ctx.df["close"], self.z_len)
        strong = volatility >= high_threshold
        calm = volatility <= low_threshold

        # Volatilite haute : on SUIT le mouvement. Volatilite basse : on le FADE.
        # C'est le meme indicateur qui produit deux paris opposes selon l'etat.
        raw = np.where(
            strong & (z.abs() >= 1.0),
            np.sign(z),
            np.where(calm & (z.abs() >= 1.5), -np.sign(z), 0),
        )
        signal = pd.Series(raw, index=index, name="signal").fillna(0).astype(int)
        return signal.where(signal != signal.shift(1, fill_value=0), 0)


class MultiDayMomentum(AlphaFamily):
    """La tendance a l'echelle de plusieurs jours se poursuit-elle ?

    Tout le reste du catalogue est intraday. Cette famille change d'horizon : elle lit
    la structure a l'echelle de la journee de trading et vise une detention longue.
    """

    name = "momentum_multijour"
    question = (
        "Le classement des rendements sur plusieurs journees de trading predit-il la "
        "journee suivante, a un horizon plus long que tout le reste du catalogue ?"
    )

    def __init__(
        self, lookback_days: int = 5, z_threshold: float = 1.0, entry_hour: int = 14
    ) -> None:
        self.lookback_days = lookback_days
        self.z_threshold = z_threshold
        self.entry_hour = entry_hour

    def parameters(self) -> dict[str, Any]:
        return {
            "jours_de_lecture": self.lookback_days,
            "seuil_z": self.z_threshold,
            "heure_entree": self.entry_hour,
        }

    def signal(self, ctx: Context) -> pd.Series:
        index = pd.DatetimeIndex(ctx.df.index)
        day = sessions.trading_day(index)

        # Cloture de la derniere journee TERMINEE : disponible des la premiere barre
        # de la journee suivante, jamais avant.
        daily_close = ctx.df["close"].groupby(day).last()
        previous = daily_close.shift(1)
        change = np.log(previous).diff(self.lookback_days)
        z = (change - change.rolling(60).mean()) / change.rolling(60).std(ddof=1)

        aligned = day.map(z).astype(float)
        fire = pd.Series(index.hour == self.entry_hour, index=index)
        raw = np.where(
            fire & (aligned >= self.z_threshold),
            1,
            np.where(fire & (aligned <= -self.z_threshold), -1, 0),
        )
        return pd.Series(raw, index=index, name="signal")


class IntradaySeasonality(AlphaFamily):
    """Certaines heures de certains jours ont-elles un biais persistant ?"""

    name = "saisonnalite_intraday"
    question = (
        "Le rendement moyen historique d'une cellule (heure UTC x jour de semaine), "
        "estime uniquement sur le passe, predit-il le rendement de cette cellule ?"
    )

    def __init__(self, min_observations: int = 60, threshold_bps: float = 2.0) -> None:
        self.min_observations = min_observations
        self.threshold_bps = threshold_bps

    def parameters(self) -> dict[str, Any]:
        return {
            "observations_min": self.min_observations,
            "seuil_bps": self.threshold_bps,
        }

    def signal(self, ctx: Context) -> pd.Series:
        index = pd.DatetimeIndex(ctx.df.index)
        close = ctx.df["close"]

        # Rendement de la barre PRECEDENTE vers la barre courante : connu en t.
        realised = np.log(close).diff()

        cell = pd.Series(
            [f"{h}-{d}" for h, d in zip(index.hour, index.dayofweek, strict=True)], index=index
        )
        grouped = realised.groupby(cell)
        # Moyenne expansive de la cellule : n'utilise que des observations deja
        # realisees. C'est ce qui distingue cette famille d'un simple "on regarde ce
        # qui a marche dans le passe" fait sur l'historique complet.
        mean_so_far = grouped.transform(lambda s: s.expanding().mean())
        count_so_far = grouped.transform(lambda s: s.expanding().count())

        threshold = self.threshold_bps / 10_000.0
        eligible = count_so_far >= self.min_observations
        raw = np.where(
            eligible & (mean_so_far >= threshold),
            1,
            np.where(eligible & (mean_so_far <= -threshold), -1, 0),
        )
        signal = pd.Series(raw, index=index, name="signal")
        return signal.where(signal != signal.shift(1, fill_value=0), 0)
