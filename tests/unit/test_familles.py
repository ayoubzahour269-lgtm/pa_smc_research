"""Contrat et semantique des familles d'hypotheses."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphalab.alpha.base import GEOMETRY, Context, build_features
from alphalab.alpha.families import crossasset, reversion, structure
from alphalab.alpha.registry import all_families, core_families, crossasset_families, family_label
from alphalab.data.costs import CostModel
from alphalab.features.causality import synthetic_ohlcv

DF = synthetic_ohlcv(2000, seed=21, freq="1h")
PEER = synthetic_ohlcv(2000, seed=22, freq="1h", start_price=15000.0)
GEO = GEOMETRY["H1"]


def _context(df: pd.DataFrame = DF, peers: dict[str, pd.DataFrame] | None = None) -> Context:
    index = pd.DatetimeIndex(df.index)
    return Context(
        symbol="TEST",
        timeframe="H1",
        df=df,
        features=build_features(df, GEO),
        cost=CostModel("TEST", "H1", pd.Series(0.3, index=index, name="spread"), "synthetique"),
        geometry=GEO,
        peers=peers if peers is not None else {"PEER": PEER},
    )


FAMILIES = all_families(["PEER"])


# --------------------------------------------------------------------------------------
# Contrat commun
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("family", FAMILIES, ids=[family_label(f) for f in FAMILIES])
def test_signal_respecte_le_contrat(family) -> None:
    """Valeurs dans {-1, 0, +1}, index aligne, aucun NaN."""
    signal = family.signal(_context())
    assert len(signal) == len(DF)
    assert signal.index.equals(DF.index)
    assert not signal.isna().any()
    assert set(np.unique(signal.to_numpy())) <= {-1, 0, 1}


@pytest.mark.parametrize("family", FAMILIES, ids=[family_label(f) for f in FAMILIES])
def test_les_parametres_sont_journalisables(family) -> None:
    """Le journal des essais doit pouvoir serialiser chaque configuration."""
    import json

    params = family.parameters()
    assert isinstance(params, dict)
    json.dumps(params)  # leve si un parametre n'est pas serialisable


def test_les_noms_de_familles_sont_uniques() -> None:
    """Deux familles homonymes se melangeraient dans le journal des essais."""
    labels = [family_label(f) for f in FAMILIES]
    assert len(labels) == len(set(labels))


def test_le_catalogue_couvre_les_themes_annonces() -> None:
    noms = {f.name for f in core_families()}
    attendus = (
        "seance_derive",
        "opening_range",
        "cassure",
        "smc_choch",
        "regime_volatilite",
        "momentum_multijour",
        "saisonnalite_intraday",
        "ensemble",
    )
    for attendu in attendus:
        assert attendu in noms


def test_les_variantes_de_cassure_sont_declarees_distinctement() -> None:
    """Un declencheur unique, des filtres en parametres : chaque variante reste un essai."""
    labels = [family_label(f) for f in core_families() if f.name == "cassure"]
    assert len(labels) == len(set(labels)) >= 3
    assert all(lbl.startswith("cassure[") for lbl in labels)


def test_sans_pair_les_familles_inter_actifs_sont_absentes() -> None:
    """Il ne faut pas laisser croire qu'une piste a ete testee alors qu'elle manquait."""
    assert crossasset_families([]) == []
    assert len(crossasset_families(["EURUSD"])) == 4


# --------------------------------------------------------------------------------------
# Semantique : le sens du signal doit etre celui annonce
# --------------------------------------------------------------------------------------


def test_le_balayage_du_haut_declenche_une_vente() -> None:
    """Un balayage prend la liquidite puis rejette : c'est un signal contraire."""
    index = pd.date_range("2021-06-01 00:00", periods=72, freq="h", tz="UTC")
    # La barre 61 tombe a 13h UTC, dans la seance de New York, et appartient a une
    # journee de trading dont la veille porte un plus-haut a 110.
    sweep_bar = 61
    rows = []
    for i in range(72):
        if i == sweep_bar:  # perce le plus-haut de la veille puis cloture en deca
            rows.append((100.0, 130.0, 99.0, 100.0))
        elif i < 24:
            rows.append((100.0, 110.0, 90.0, 100.0))
        else:
            rows.append((100.0, 101.0, 99.0, 100.0))
    df = pd.DataFrame(
        {
            "open": [r[0] for r in rows],
            "high": [r[1] for r in rows],
            "low": [r[2] for r in rows],
            "close": [r[3] for r in rows],
            "volume": 1.0,
        },
        index=index,
    )
    signal = structure.LiquiditySweepReversal(min_wick_atr=0.0, session="newyork").signal(
        _context(df, peers={})
    )
    assert index[sweep_bar].hour == 13
    assert signal.iloc[sweep_bar] == -1


def test_le_gap_haussier_est_vendu_en_mode_comblement() -> None:
    index = pd.date_range("2021-06-01 22:00", periods=48, freq="h", tz="UTC")
    close = np.full(48, 100.0)
    open_ = np.full(48, 100.0)
    open_[24] = 130.0  # premiere barre de la journee suivante, gap haussier
    close[24] = 130.0
    df = pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) + 1.0,
            "low": np.minimum(open_, close) - 1.0,
            "close": close,
            "volume": 1.0,
        },
        index=index,
    )
    fade = reversion.OvernightGap(min_atr=0.1, fade=True).signal(_context(df, peers={}))
    suivre = reversion.OvernightGap(min_atr=0.1, fade=False).signal(_context(df, peers={}))
    assert fade.iloc[24] == -1
    assert suivre.iloc[24] == 1


def test_le_lead_lag_inverse_bien_son_sens() -> None:
    """Les deux sens sont testes separement : aucun n'est suppose juste a l'avance."""
    ctx = _context()
    suivre = crossasset.LeadLag("PEER", sign=1).signal(ctx)
    opposer = crossasset.LeadLag("PEER", sign=-1).signal(ctx)
    assert (suivre == -opposer).all()
    assert int((suivre != 0).sum()) > 0


def test_famille_inter_actifs_sans_donnees_reste_muette() -> None:
    """Sans le pair, la famille ne doit rien inventer."""
    signal = crossasset.CorrelationDivergence("EURUSD").signal(_context(peers={}))
    assert int((signal != 0).sum()) == 0


# --------------------------------------------------------------------------------------
# Conversion en ordres
# --------------------------------------------------------------------------------------


def test_le_signal_produit_une_entree_a_la_barre_suivante() -> None:
    ctx = _context()
    signal = pd.Series(0, index=DF.index)
    signal.iloc[500] = 1
    orders = ctx.orders_from_signal(signal, tag="essai")
    assert len(orders) == 1
    assert orders[0].signal_ts == DF.index[500]
    assert orders[0].direction == 1
    atr = float(ctx.atr.iloc[500])
    assert orders[0].stop_distance == pytest.approx(GEO.sl_atr * atr)
    assert orders[0].target_distance == pytest.approx(GEO.tp_atr * atr)


def test_un_signal_en_periode_de_chauffe_est_ignore() -> None:
    """Sans ATR defini, la geometrie de risque n'existe pas : pas d'ordre."""
    ctx = _context()
    signal = pd.Series(0, index=DF.index)
    signal.iloc[2] = 1
    assert ctx.orders_from_signal(signal, tag="essai") == []
