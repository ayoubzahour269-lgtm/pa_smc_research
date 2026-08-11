"""Chaque famille doit produire un signal causal.

Une famille qui anticipe produirait des ordres impossibles a passer en reel, et un
backtest flatteur. Le controle porte sur le SIGNAL, en amont de la conversion en
ordres : c'est la que se trouve toute la logique propre a la famille.
"""

from __future__ import annotations

import pandas as pd
import pytest

from alphalab.alpha.base import GEOMETRY, Context, build_features
from alphalab.alpha.registry import all_families, family_label
from alphalab.data.costs import CostModel
from alphalab.features.causality import assert_causal, synthetic_ohlcv

DF = synthetic_ohlcv(3000, seed=11, freq="1h")
PEER = synthetic_ohlcv(3000, seed=12, freq="1h", start_price=15000.0)
GEO = GEOMETRY["H1"]


def _context(df: pd.DataFrame) -> Context:
    """Contexte reconstruit sur la trame fournie — y compris tronquee.

    Point cle : les features, le cout et les donnees du pair sont recalcules a partir
    de `df`. Fournir des features calculees sur l'historique complet masquerait
    justement l'anticipation qu'on cherche a detecter.
    """
    index = pd.DatetimeIndex(df.index)
    spread = pd.Series(0.3, index=index, name="spread")
    return Context(
        symbol="TEST",
        timeframe="H1",
        df=df,
        features=build_features(df, GEO),
        cost=CostModel("TEST", "H1", spread, "synthetique"),
        geometry=GEO,
        peers={"PEER": PEER.loc[: index[-1]]},
    )


FAMILIES = all_families(["PEER"])


@pytest.mark.parametrize("family", FAMILIES, ids=[family_label(f) for f in FAMILIES])
def test_signal_de_famille_causal(family) -> None:
    assert_causal(
        lambda d, f=family: f.signal(_context(d)).to_frame(),
        DF,
        label=family_label(family),
        n_probes=12,
        min_history=600,
    )


def test_le_spread_variable_ne_fait_pas_anticiper_la_microstructure() -> None:
    """Les familles de microstructure lisent le cout : leur quantile doit rester glissant.

    Un quantile calcule sur tout l'historique de spread ferait entrer le futur dans la
    decision — c'est le piege propre a ces deux familles.
    """
    from alphalab.alpha.families import microstructure

    rng = pd.Series(
        pd.Series(range(len(DF)))
        .apply(lambda i: 0.2 + 0.4 * ((i * 7919) % 97) / 97)
        .to_numpy(),
        index=DF.index,
        name="spread",
    )

    def ctx_with_spread(d: pd.DataFrame) -> Context:
        ctx = _context(d)
        ctx.cost = CostModel("TEST", "H1", rng.loc[: d.index[-1]], "synthetique variable")
        return ctx

    for family in (
        microstructure.LiquidityRegimeBreakout(),
        microstructure.SpreadShockFade(),
    ):
        assert_causal(
            lambda d, f=family: f.signal(ctx_with_spread(d)).to_frame(),
            DF,
            label=family_label(family),
            n_probes=10,
            min_history=700,
        )
