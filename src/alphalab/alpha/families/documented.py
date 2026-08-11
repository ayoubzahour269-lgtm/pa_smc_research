"""Anomalies documentees dans la litterature, transposees a notre cadre.

Toutes les familles precedentes venaient de l'analyse technique praticienne. Celles-ci
viennent de travaux qui ont ete publies, repliques et discutes. Cela ne les rend pas
vraies chez nous : une anomalie mesuree sur des actions americaines en journalier, sur
une periode donnee, n'a aucune obligation de survivre sur un CFD Nasdaq en horaire,
apres spread reel, sur 2019-2022. C'est precisement ce qu'on mesure.

Un avantage decisif de ces pistes : leurs definitions et leurs parametres viennent de
l'exterieur. Ils n'ont pas ete choisis en regardant nos donnees, ce qui elimine une
source de sur-ajustement que les familles maison ne peuvent jamais totalement ecarter.

Sources d'inspiration :
  - momentum intraday : Gao, Han, Li et Zhou, "Market intraday momentum" (2018) — sur
    les indices actions, le rendement de la premiere demi-heure predit celui de la
    derniere ;
  - drift overnight : Lou, Polk et Skouras, "A tug of war" (2019) — sur les indices, le
    rendement s'accumule hors seance plutot que pendant ;
  - momentum de series temporelles : Moskowitz, Ooi et Pedersen (2012) — le signe du
    rendement passe predit le suivant, sur toutes les classes d'actifs ;
  - agregation autour des chiffres ronds : Osler (2003) — les ordres stop se
    concentrent juste au-dela des niveaux ronds, les prises de benefice dessus.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from alphalab.alpha.base import AlphaFamily, Context
from alphalab.features import indicators, sessions


class IntradayMomentum(AlphaFamily):
    """Le debut de seance predit-il la fin de seance ?"""

    name = "momentum_intraseance"
    question = (
        "Le rendement de la premiere heure de la seance americaine predit-il celui de "
        "la derniere heure de cette meme seance ?"
    )

    def __init__(
        self, first_hour: int = 14, trade_hour: int = 20, min_move_atr: float = 0.3
    ) -> None:
        self.first_hour = first_hour
        self.trade_hour = trade_hour
        self.min_move_atr = min_move_atr

    def parameters(self) -> dict[str, Any]:
        return {
            "heure_reference": self.first_hour,
            "heure_entree": self.trade_hour,
            "mouvement_min_atr": self.min_move_atr,
        }

    def signal(self, ctx: Context) -> pd.Series:
        index = pd.DatetimeIndex(ctx.df.index)
        day = sessions.trading_day(index)

        # Rendement de la barre de reference, exprime en ATR pour rester comparable
        # entre instruments et entre regimes de volatilite.
        move = (ctx.df["close"] - ctx.df["open"]) / ctx.features["atr"]
        reference = move.where(pd.Series(index.hour == self.first_hour, index=index))
        # `ffill` a l'interieur de la journee : la valeur est connue des la cloture de
        # la barre de reference, donc bien avant l'heure d'entree.
        known = reference.groupby(day).ffill()

        fire = pd.Series(index.hour == self.trade_hour, index=index)
        raw = np.where(
            fire & (known >= self.min_move_atr),
            1,
            np.where(fire & (known <= -self.min_move_atr), -1, 0),
        )
        return pd.Series(raw, index=index, name="signal")


class OvernightDrift(AlphaFamily):
    """Le rendement s'accumule-t-il hors seance plutot que pendant ?"""

    name = "drift_overnight"
    question = (
        "Acheter a la fin de la seance liquide et sortir a la reprise capture-t-il un "
        "rendement systematiquement different de celui de la seance elle-meme ?"
    )

    def __init__(self, entry_hour: int = 20, exit_hour: int = 14, direction: int = 1) -> None:
        self.entry_hour = entry_hour
        self.exit_hour = exit_hour
        self.direction = direction

    @property
    def name_with_peer(self) -> str:
        return f"{self.name}[{'achat' if self.direction == 1 else 'vente'}]"

    def parameters(self) -> dict[str, Any]:
        return {
            "heure_entree": self.entry_hour,
            "heure_sortie": self.exit_hour,
            "sens": self.direction,
        }

    def signal(self, ctx: Context) -> pd.Series:
        index = pd.DatetimeIndex(ctx.df.index)
        fire = pd.Series(index.hour == self.entry_hour, index=index)
        raw = np.where(fire, self.direction, 0)
        return pd.Series(raw, index=index, name="signal")


class FailedBreakout(AlphaFamily):
    """Une cassure qui ne tient pas se retourne-t-elle ?

    Distincte du balayage de liquidite, qui se joue a l'interieur d'une seule barre
    (meche au-dela, cloture en deca). Ici la cassure a bel et bien eu lieu — cloture
    au-dela de l'extreme — et c'est son ECHEC a se maintenir sur les barres suivantes
    qui constitue le signal.
    """

    name = "cassure_ratee"
    question = (
        "Lorsqu'une cloture depasse un extreme recent puis repasse en deca dans les "
        "barres qui suivent, le mouvement se retourne-t-il dans le sens inverse ?"
    )

    def __init__(self, lookback: int = 24, max_bars: int = 6) -> None:
        self.lookback = lookback
        self.max_bars = max_bars

    def parameters(self) -> dict[str, Any]:
        return {"lookback": self.lookback, "barres_max": self.max_bars}

    def signal(self, ctx: Context) -> pd.Series:
        index = pd.DatetimeIndex(ctx.df.index)
        close = ctx.df["close"]
        highest = indicators.rolling_high(ctx.df, self.lookback)
        lowest = indicators.rolling_low(ctx.df, self.lookback)

        broke_up = (close > highest) & ~(close > highest).shift(1, fill_value=False)
        broke_down = (close < lowest) & ~(close < lowest).shift(1, fill_value=False)

        # Niveau franchi et age de la cassure, calcules uniquement vers le passe.
        level_up = highest.where(broke_up).ffill()
        level_down = lowest.where(broke_down).ffill()
        age_up = _bars_since(broke_up)
        age_down = _bars_since(broke_down)

        failed_up = age_up.between(1, self.max_bars) & (close < level_up)
        failed_down = age_down.between(1, self.max_bars) & (close > level_down)

        raw = np.where(failed_up, -1, np.where(failed_down, 1, 0))
        signal = pd.Series(raw, index=index, name="signal")
        return signal.where(signal != signal.shift(1, fill_value=0), 0)


class RoundNumber(AlphaFamily):
    """Les chiffres ronds agissent-ils comme des aimants ou comme des barrieres ?"""

    name = "niveaux_ronds"
    question = (
        "A l'approche d'un niveau rond, le prix est-il repousse — les ordres de prise "
        "de benefice s'y concentrant — ou traverse-t-il en accelerant, les stops etant "
        "places juste au-dela ?"
    )

    def __init__(
        self, step_frac: float = 0.01, proximity_atr: float = 0.25, fade: bool = True
    ) -> None:
        # Pas exprime en fraction du prix : 1 % donne ~20 points sur un or a 2000 et
        # ~150 sur un Nasdaq a 15000, ce qui correspond bien aux niveaux que les
        # operateurs regardent sur chaque instrument.
        self.step_frac = step_frac
        self.proximity_atr = proximity_atr
        self.fade = fade

    @property
    def name_with_peer(self) -> str:
        return f"{self.name}[{'rejet' if self.fade else 'cassure'}]"

    def parameters(self) -> dict[str, Any]:
        return {
            "pas_fraction_du_prix": self.step_frac,
            "proximite_atr": self.proximity_atr,
            "sens": "rejet" if self.fade else "cassure",
        }

    def signal(self, ctx: Context) -> pd.Series:
        index = pd.DatetimeIndex(ctx.df.index)
        close = ctx.df["close"]
        atr = ctx.features["atr"]

        # Le pas est arrondi a une puissance de dix "propre" pour tomber sur des
        # niveaux que les operateurs regardent reellement (50, 100, 500...).
        rough = close.shift(1) * self.step_frac
        step = 10 ** np.floor(np.log10(rough.where(rough > 0)))
        nearest = (close / step).round() * step
        distance = (close - nearest).abs()

        near = distance <= self.proximity_atr * atr
        approaching_from_below = close < nearest
        approaching_from_above = close > nearest

        if self.fade:
            # Rejet : on parie que le niveau tient.
            raw = np.where(
                near & approaching_from_below, -1, np.where(near & approaching_from_above, 1, 0)
            )
        else:
            # Cassure : on parie que le niveau cede, les stops accelerant le mouvement.
            raw = np.where(
                near & approaching_from_below, 1, np.where(near & approaching_from_above, -1, 0)
            )
        signal = pd.Series(raw, index=index, name="signal")
        return signal.where(signal != signal.shift(1, fill_value=0), 0)


class TimeSeriesMomentum(AlphaFamily):
    """Le signe du rendement passe predit-il le suivant, a horizon long ?

    Le resultat le plus replique de la litterature sur le momentum, mesure a l'origine
    sur des horizons de 1 a 12 mois et sur des dizaines de marches. Notre fenetre
    in-sample (4 a 8 ans) permet de tester le bas de cette plage.
    """

    name = "momentum_series_temporelles"
    question = (
        "Le signe du rendement des N derniers jours predit-il le rendement suivant, "
        "comme le documente la litterature sur les series temporelles ?"
    )

    def __init__(self, lookback_days: int = 60, entry_hour: int = 14) -> None:
        self.lookback_days = lookback_days
        self.entry_hour = entry_hour

    def parameters(self) -> dict[str, Any]:
        return {"jours_de_lecture": self.lookback_days, "heure_entree": self.entry_hour}

    def signal(self, ctx: Context) -> pd.Series:
        index = pd.DatetimeIndex(ctx.df.index)
        day = sessions.trading_day(index)

        # Cloture de la derniere journee TERMINEE : connue des la journee suivante.
        daily_close = ctx.df["close"].groupby(day).last().shift(1)
        change = np.log(daily_close).diff(self.lookback_days)

        aligned = day.map(change).astype(float)
        fire = pd.Series(index.hour == self.entry_hour, index=index)
        raw = np.where(fire & (aligned > 0), 1, np.where(fire & (aligned < 0), -1, 0))
        return pd.Series(raw, index=index, name="signal")


def _bars_since(flag: pd.Series) -> pd.Series:
    """Nombre de barres ecoulees depuis le dernier `True`. NaN si jamais vu."""
    positions = pd.Series(np.arange(len(flag), dtype=float), index=flag.index)
    last = positions.where(flag.fillna(False)).ffill()
    return positions - last
