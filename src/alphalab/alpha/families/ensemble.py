"""Ensemble : trader uniquement quand plusieurs mecanismes independants s'accordent.

L'intuition est repandue — "plus il y a de confluence, meilleur est le signal" — et
elle est fausse telle quelle. Faire voter des familles redondantes n'apporte rien :
quatre variantes d'une cassure qui s'accordent a 100 % du temps forment un vote a
l'unanimite qui ne contient qu'une seule opinion.

Cette famille ne fait donc voter que des mecanismes dont la mesure de diversite montre
qu'ils sont reellement distincts. C'est la seule version du vote qui teste quelque
chose : un accord entre approches independantes est un evenement informatif, un accord
entre variantes du meme indicateur ne l'est pas.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from alphalab.alpha.base import AlphaFamily, Context


class Consensus(AlphaFamily):
    """L'accord de plusieurs mecanismes independants vaut-il mieux que chacun seul ?"""

    name = "ensemble"
    question = (
        "Lorsque plusieurs familles issues de mecanismes distincts pointent dans le "
        "meme sens sur une fenetre courte, ce consensus est-il plus fiable que chaque "
        "famille prise isolement ?"
    )

    def __init__(
        self,
        members: Sequence[AlphaFamily] | None = None,
        min_votes: int = 2,
        window: int = 3,
    ) -> None:
        self._members = list(members) if members is not None else None
        self.min_votes = min_votes
        self.window = window

    def members(self) -> list[AlphaFamily]:
        """Membres du vote, choisis dans des groupes de mecanismes differents.

        Import differe : le registre importe ce module, l'inverse creerait un cycle.
        """
        if self._members is not None:
            return self._members
        from alphalab.alpha.families import breakout, regime, reversion, structure

        return [
            breakout.Breakout(condition="adx"),
            reversion.VwapReversion(),
            structure.LiquiditySweepReversal(),
            regime.VolatilityRegimeSwitch(),
        ]

    def parameters(self) -> dict[str, Any]:
        return {
            "membres": [m.name for m in self.members()],
            "votes_min": self.min_votes,
            "fenetre_vote": self.window,
        }

    def signal(self, ctx: Context) -> pd.Series:
        index = pd.DatetimeIndex(ctx.df.index)
        members = self.members()
        if not members:
            return pd.Series(0, index=index, name="signal")

        # Les familles se declenchent rarement sur la meme barre exacte : on accorde
        # une fenetre de quelques barres pour qu'un accord reste detectable, sans
        # jamais regarder vers l'avant.
        votes = pd.DataFrame(
            {
                m.name: m.signal(ctx)
                .reindex(index)
                .fillna(0)
                .rolling(self.window, min_periods=1)
                .sum()
                .clip(-1, 1)
                for m in members
            },
            index=index,
        )
        longs = (votes > 0).sum(axis=1)
        shorts = (votes < 0).sum(axis=1)

        # Un consensus n'en est un que s'il est net : on exige la majorite requise ET
        # l'absence de vote contraire.
        raw = np.where(
            (longs >= self.min_votes) & (shorts == 0),
            1,
            np.where((shorts >= self.min_votes) & (longs == 0), -1, 0),
        )
        signal = pd.Series(raw, index=index, name="signal")
        return signal.where(signal != signal.shift(1, fill_value=0), 0)
