"""Controle de causalite sur les donnees de marche reelles.

Les tests sur donnees synthetiques suffisent en principe — la causalite est une
propriete du code, pas des donnees. Mais les vraies series contiennent ce que les
marches aleatoires n'ont pas : week-ends, jours feries, trous de plusieurs jours,
egalites exactes de prix, barres de volume nul. Ces irregularites empruntent des
branches de code que la marche aleatoire ne visite jamais.
"""

from __future__ import annotations

import pandas as pd
import pytest

from alphalab.data import registry, snapshot
from alphalab.features import resample, sessions, smc
from alphalab.features.causality import assert_causal

pytestmark = pytest.mark.slow

#: Tranche in-sample, assez longue pour couvrir des week-ends et des feries, assez
#: courte pour que le rejeu point-dans-le-temps reste rapide.
WINDOW = (pd.Timestamp("2021-01-01", tz="UTC"), pd.Timestamp("2021-07-01", tz="UTC"))


def _slice(symbol: str, timeframe: str) -> pd.DataFrame:
    df = snapshot.load(registry.require(symbol, timeframe, "BID"), copy=False)
    start, end = WINDOW
    return df[(df.index >= start) & (df.index < end)]


@pytest.mark.parametrize("symbol", ["XAUUSD", "NAS100"])
def test_smc_causal_sur_donnees_reelles(symbol: str) -> None:
    assert_causal(
        smc.build_all, _slice(symbol, "H1"), label=f"smc.build_all({symbol})", n_probes=12
    )


@pytest.mark.parametrize("symbol", ["XAUUSD", "NAS100"])
@pytest.mark.parametrize("window", sessions.SESSIONS)
def test_ranges_de_seance_causaux_sur_donnees_reelles(
    symbol: str, window: sessions.SessionWindow
) -> None:
    """Les trous de week-end coupent les occurrences de seance : cas absent du synthetique."""
    assert_causal(
        lambda d, w=window: sessions.completed_window_range(d, w),
        _slice(symbol, "H1"),
        label=f"{window.name}({symbol})",
        n_probes=12,
    )


@pytest.mark.parametrize("symbol", ["XAUUSD", "NAS100"])
def test_contexte_superieur_causal_sur_donnees_reelles(symbol: str) -> None:
    assert_causal(
        resample.multi_timeframe_context,
        _slice(symbol, "H1"),
        label=f"multi_timeframe_context({symbol})",
        n_probes=10,
    )


def test_causalite_tenue_sur_m15_malgre_les_trous() -> None:
    """M15 sur l'or : ~17 000 barres sur six mois, avec un trou hebdomadaire."""
    assert_causal(smc.build_all, _slice("XAUUSD", "M15"), label="smc M15", n_probes=6)
