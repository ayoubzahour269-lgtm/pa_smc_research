"""Famille de microstructure : le spread comme signal, pas seulement comme cout.

Le spread est traite partout ailleurs dans la plateforme comme une charge a payer. Il
porte pourtant une information : il s'elargit quand les teneurs de marche se protegent,
et se resserre quand la liquidite est abondante. La question est de savoir si ce regime
de liquidite precede quoi que ce soit d'exploitable.

Precaution indispensable ici : le spread etant aussi le cout, une famille qui le prend
pour signal risque de ne decouvrir que "trader quand c'est moins cher est moins cher".
Le temoin apparie a l'heure du protocole neutralise en grande partie ce biais, puisque
le profil de spread est fortement horaire. La porte de sensibilite au cout (R > 0 a
1,5x le spread) acheve le controle.

Ce module ne contient plus qu'une famille : la cassure en regime de liquidite
abondante partageait son declencheur avec les autres cassures et vit desormais dans
`families/breakout.py` sous le filtre `spread_bas`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from alphalab.alpha.base import AlphaFamily, Context
from alphalab.features import indicators


class SpreadShockFade(AlphaFamily):
    """Un elargissement brutal du spread precede-t-il un retournement ?"""

    name = "microstructure_choc"
    question = (
        "Un elargissement soudain du spread, signe de stress des teneurs de marche, "
        "annonce-t-il un retournement du mouvement en cours ?"
    )

    def __init__(self, z_threshold: float = 3.0, window: int = 480) -> None:
        self.z_threshold = z_threshold
        self.window = window

    def parameters(self) -> dict[str, Any]:
        return {"seuil_z": self.z_threshold, "fenetre": self.window}

    def signal(self, ctx: Context) -> pd.Series:
        index = pd.DatetimeIndex(ctx.df.index)
        spread = ctx.cost.spread.reindex(index)
        z = indicators.zscore(spread, self.window)
        shock = z >= self.z_threshold

        # Sens du mouvement de la barre : on parie contre lui.
        move = np.sign(ctx.df["close"] - ctx.df["open"])
        raw = np.where(shock, -move, 0)
        signal = pd.Series(raw, index=index, name="signal").fillna(0).astype(int)
        return signal.where(signal != signal.shift(1, fill_value=0), 0)
