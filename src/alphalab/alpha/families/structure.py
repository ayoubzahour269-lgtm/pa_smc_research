"""Familles de structure de prix (SMC).

Aucun statut privilegie : ces familles passent exactement le meme protocole que les
onze autres. C'est une difference de fond avec l'orientation initiale de ce depot, qui
designait la branche SMC comme la suite naturelle du travail. Ici elle est une
hypothese parmi d'autres, et elle sera jugee sur les memes chiffres.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from alphalab.alpha.base import AlphaFamily, Context
from alphalab.features import sessions, smc


class ChochRetracement(AlphaFamily):
    """Apres un changement de caractere, le retracement offre-t-il une entree ?"""

    name = "smc_choch"
    question = (
        "Apres un CHoCH — premiere cassure contraire a la structure en place — un "
        "retour du prix dans la moitie favorable de la jambe offre-t-il une entree "
        "dans le nouveau sens ?"
    )

    def __init__(self, width: int = 3, max_wait: int = 24) -> None:
        self.width = width
        self.max_wait = max_wait

    def parameters(self) -> dict[str, Any]:
        return {"largeur_pivot": self.width, "attente_max": self.max_wait}

    def signal(self, ctx: Context) -> pd.Series:
        """Entree au RETRACEMENT, pas sur la barre de cassure.

        Point de definition essentiel : un CHoCH haussier se produit lorsque la
        cloture depasse le dernier sommet confirme — il est donc, par construction,
        toujours en premium. Exiger d'acheter "en discount sur la barre de CHoCH"
        est contradictoire et ne produit jamais aucun signal. La regle reelle de la
        methode est d'attendre que le prix revienne sous l'equilibre de la jambe.
        """
        struct = smc.structure(ctx.df, self.width)
        zones = smc.premium_discount(ctx.df, self.width)

        choch = struct["choch"].to_numpy()
        state = struct["structure"].to_numpy()
        equilibrium = zones["equilibrium"].to_numpy()
        close = ctx.df["close"].to_numpy()

        n = len(ctx.df)
        out = np.zeros(n, dtype=int)
        pending_dir = 0
        pending_eq = np.nan
        waited = 0

        for i in range(n):
            if choch[i]:
                # Un nouveau CHoCH remplace le precedent : on ne cumule pas les attentes.
                pending_dir = int(state[i])
                pending_eq = equilibrium[i]
                waited = 0
                continue
            if pending_dir == 0:
                continue
            waited += 1
            if waited > self.max_wait or not np.isfinite(pending_eq):
                pending_dir = 0
                continue
            reached = close[i] <= pending_eq if pending_dir == 1 else close[i] >= pending_eq
            if reached:
                out[i] = pending_dir
                pending_dir = 0

        return pd.Series(out, index=ctx.df.index, name="signal")


class BosContinuation(AlphaFamily):
    """Une cassure de structure dans le sens etabli se poursuit-elle ?"""

    name = "smc_bos"
    question = (
        "Un BOS — cassure confirmant la structure en cours — marque-t-il une "
        "continuation exploitable, distincte du CHoCH ?"
    )

    def __init__(self, width: int = 3, session: str = "newyork") -> None:
        self.width = width
        self.session = session

    def parameters(self) -> dict[str, Any]:
        return {"largeur_pivot": self.width, "seance": self.session}

    def signal(self, ctx: Context) -> pd.Series:
        index = pd.DatetimeIndex(ctx.df.index)
        struct = smc.structure(ctx.df, self.width)
        window = next(w for w in sessions.SESSIONS if w.name == self.session).mask(index)
        direction = struct["structure"].where(struct["bos"] & window, 0)
        return direction.fillna(0).astype(int).rename("signal")


class LiquiditySweepReversal(AlphaFamily):
    """Une chasse aux stops sur les extremes de la veille se retourne-t-elle ?"""

    name = "smc_balayage"
    question = (
        "Lorsque le prix perce un extreme de la journee precedente puis cloture en "
        "deca dans la meme barre, se retourne-t-il dans le sens du rejet ?"
    )

    def __init__(self, min_wick_atr: float = 0.15, session: str = "newyork") -> None:
        self.min_wick_atr = min_wick_atr
        self.session = session

    def parameters(self) -> dict[str, Any]:
        return {"meche_min_atr": self.min_wick_atr, "seance": self.session}

    def signal(self, ctx: Context) -> pd.Series:
        index = pd.DatetimeIndex(ctx.df.index)
        levels = sessions.previous_day_levels(ctx.df)
        sweeps = smc.liquidity_sweeps(
            ctx.df, levels["pdh"], levels["pdl"], min_wick_atr=self.min_wick_atr
        )
        window = next(w for w in sessions.SESSIONS if w.name == self.session).mask(index)
        # Un balayage du haut est un signal de VENTE : la liquidite a ete prise puis
        # rejetee. C'est ce qui le distingue d'une cassure, ou l'on achete.
        raw = np.where(
            sweeps["sweep_high"] & window, -1, np.where(sweeps["sweep_low"] & window, 1, 0)
        )
        return pd.Series(raw, index=index, name="signal")


class FvgRetest(AlphaFamily):
    """Le retour du prix dans un ecart de valeur recent offre-t-il une entree ?"""

    name = "smc_fvg"
    question = (
        "Apres la formation d'un ecart de valeur significatif, le premier retour du "
        "prix dans cet ecart repart-il dans le sens de l'impulsion qui l'a cree ?"
    )

    def __init__(self, min_atr: float = 0.3, max_age: int = 12) -> None:
        self.min_atr = min_atr
        self.max_age = max_age

    def parameters(self) -> dict[str, Any]:
        return {"fvg_min_atr": self.min_atr, "age_max": self.max_age}

    def signal(self, ctx: Context) -> pd.Series:
        index = pd.DatetimeIndex(ctx.df.index)
        fvg = smc.fair_value_gaps(ctx.df, min_atr=self.min_atr)

        # Dernier FVG haussier connu et son age, calcules uniquement vers le passe.
        bull_top = fvg["fvg_bull_bottom"].ffill()
        bull_age = _bars_since(fvg["fvg_bull"])
        bear_bottom = fvg["fvg_bear_top"].ffill()
        bear_age = _bars_since(fvg["fvg_bear"])

        # Le retest doit avoir lieu APRES la barre de formation, d'ou age >= 1.
        retest_bull = (
            bull_age.between(1, self.max_age)
            & (ctx.df["low"] <= bull_top)
            & (ctx.df["close"] > bull_top)
        )
        retest_bear = (
            (bear_age.between(1, self.max_age))
            & (ctx.df["high"] >= bear_bottom)
            & (ctx.df["close"] < bear_bottom)
        )
        raw = np.where(retest_bull, 1, np.where(retest_bear, -1, 0))
        signal = pd.Series(raw, index=index, name="signal")
        return signal.where(signal != signal.shift(1, fill_value=0), 0)


def _bars_since(flag: pd.Series) -> pd.Series:
    """Nombre de barres ecoulees depuis le dernier `True`. NaN si jamais vu."""
    positions = pd.Series(np.arange(len(flag), dtype=float), index=flag.index)
    last = positions.where(flag.fillna(False)).ffill()
    return positions - last
