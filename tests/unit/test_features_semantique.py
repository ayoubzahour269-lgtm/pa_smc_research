"""Justesse des features, sur des cas construits a la main.

La causalite (verifiee dans `tests/antilookahead/`) garantit qu'une feature ne triche
pas. Elle ne garantit pas qu'elle calcule la bonne chose. Ces tests-ci s'en chargent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from tests.conftest import flat_bars, make_frame

from alphalab.features import indicators, resample, sessions, smc

Bar = tuple[float, float, float, float]


# --------------------------------------------------------------------------------------
# Indicateurs
# --------------------------------------------------------------------------------------


def test_true_range_prend_le_maximum_des_trois_mesures() -> None:
    bars: list[Bar] = [(100, 101, 99, 100), (100, 102, 99.5, 101), (101, 101.5, 95, 96)]
    tr = indicators.true_range(make_frame(bars))
    # Barre 0 : pas de cloture precedente, donc high - low.
    assert tr.iloc[0] == pytest.approx(2.0)
    # Barre 1 : h-l = 2.5 domine |h - c_prec| = 2.0 et |l - c_prec| = 0.5.
    assert tr.iloc[1] == pytest.approx(2.5)
    # Barre 2 : h-l = 6.5 domine |l - c_prec| = 6.0 — c'est bien le maximum des trois.
    assert tr.iloc[2] == pytest.approx(6.5)


def test_atr_wilder_et_sma_different() -> None:
    """Les deux lissages doivent donner des valeurs distinctes : le choix compte."""
    # Amplitude variable : avec un range constant, les deux lissages coincident et le
    # test ne prouverait rien.
    df = make_frame(
        [(100 + i, 101 + i + (i % 7), 99 + i - (i % 5), 100.5 + i) for i in range(60)]
    )
    assert indicators.atr(df, 14).iloc[-1] != pytest.approx(
        indicators.atr(df, 14, wilder=False).iloc[-1]
    )


def test_rsi_vaut_100_en_hausse_ininterrompue() -> None:
    df = make_frame([(100 + i, 100.5 + i, 99.5 + i, 100 + i) for i in range(40)])
    assert indicators.rsi(df["close"], 14).iloc[-1] == pytest.approx(100.0)


def test_rolling_high_exclut_la_barre_courante() -> None:
    """Inclure la barre courante rendrait toute cassure triviale."""
    bars: list[Bar] = [(100, 100 + i, 99, 100) for i in range(1, 11)]
    df = make_frame(bars)
    hh = indicators.rolling_high(df, 3)
    # Les hauts valent 101..110. En position 5, les 3 barres precedentes sont les
    # positions 2, 3, 4 (hauts 103, 104, 105) : le maximum vaut donc 105.
    assert hh.iloc[5] == pytest.approx(105.0)
    assert hh.iloc[5] < df["high"].iloc[5], "la barre courante doit rester exclue"


def test_body_and_wicks_normalise_par_le_range() -> None:
    df = make_frame([(100.0, 110.0, 90.0, 105.0)])
    out = indicators.body_and_wicks(df)
    assert out["body"].iloc[0] == pytest.approx(5.0)
    assert out["upper_wick"].iloc[0] == pytest.approx(5.0)
    assert out["lower_wick"].iloc[0] == pytest.approx(10.0)
    assert out["body_frac"].iloc[0] == pytest.approx(0.25)
    assert bool(out["bullish"].iloc[0])


# --------------------------------------------------------------------------------------
# Seances
# --------------------------------------------------------------------------------------


def test_journee_de_trading_bascule_a_22h() -> None:
    index = pd.DatetimeIndex(
        ["2021-03-01 21:00", "2021-03-01 22:00", "2021-03-02 05:00"], tz="UTC"
    )
    day = sessions.trading_day(index)
    assert day.iloc[0] != day.iloc[1], "22h UTC doit ouvrir une nouvelle journee"
    assert day.iloc[1] == day.iloc[2], "22h et le lendemain 05h sont la meme journee"


def test_progression_de_seance_est_temporelle() -> None:
    index = pd.date_range("2021-03-01 22:00", periods=24, freq="h", tz="UTC")
    df = make_frame(flat_bars(24)).set_axis(index)
    prog = sessions.session_progress(df)
    assert prog.iloc[0] == pytest.approx(0.0)
    assert prog.iloc[12] == pytest.approx(0.5)
    assert prog.max() < 1.0


def test_range_de_seance_est_celui_de_l_occurrence_precedente() -> None:
    """La valeur exposee doit venir d'une occurrence CLOSE, jamais de celle en cours."""
    index = pd.date_range("2021-03-01 00:00", periods=48, freq="h", tz="UTC")
    bars: list[Bar] = []
    for i in range(48):
        # Journee 1 : asiatique entre 100 et 110. Journee 2 : entre 200 et 210.
        base = 100.0 if i < 24 else 200.0
        if index[i].hour < 7:
            bars.append((base, base + 10, base, base + 5))
        else:
            bars.append((base + 50, base + 51, base + 49, base + 50))
    df = make_frame(bars).set_axis(index)

    asia = next(w for w in sessions.SESSIONS if w.name == "asie")
    out = sessions.completed_window_range(df, asia)

    # Pendant la seance asiatique du jour 1, rien n'est encore connu.
    assert np.isnan(out["asie_high"].iloc[3])
    # Apres 07h le jour 1, on connait le range asiatique du jour 1.
    assert out["asie_high"].iloc[10] == pytest.approx(110.0)
    # Pendant la seance asiatique du jour 2, on voit encore celui du jour 1.
    assert out["asie_high"].iloc[27] == pytest.approx(110.0)
    # Apres 07h le jour 2, on bascule sur le range du jour 2.
    assert out["asie_high"].iloc[34] == pytest.approx(210.0)


def test_niveaux_de_la_veille() -> None:
    index = pd.date_range("2021-03-01 22:00", periods=48, freq="h", tz="UTC")
    bars: list[Bar] = [
        (100, 120, 80, 100) if i < 24 else (200, 220, 180, 200) for i in range(48)
    ]
    df = make_frame(bars).set_axis(index)
    prev = sessions.previous_day_levels(df)
    assert np.isnan(prev["pdh"].iloc[0]), "aucune veille disponible au premier jour"
    assert prev["pdh"].iloc[24] == pytest.approx(120.0)
    assert prev["pdl"].iloc[24] == pytest.approx(80.0)


# --------------------------------------------------------------------------------------
# Multi-timeframe
# --------------------------------------------------------------------------------------


def test_agregation_h1_depuis_m15() -> None:
    index = pd.date_range("2021-03-01 00:00", periods=8, freq="15min", tz="UTC")
    bars: list[Bar] = [(10 + i, 20 + i, 5 + i, 15 + i) for i in range(8)]
    df = make_frame(bars).set_axis(index)
    h1 = resample.to_timeframe(df, "H1")
    assert len(h1) == 2
    assert h1["open"].iloc[0] == pytest.approx(10.0)
    assert h1["high"].iloc[0] == pytest.approx(23.0)
    assert h1["low"].iloc[0] == pytest.approx(5.0)
    assert h1["close"].iloc[0] == pytest.approx(18.0)


def test_le_contexte_superieur_exclut_la_barre_en_cours() -> None:
    """La barre H1 couvrant 00:00-01:00 ne doit apparaitre qu'a partir de 01:00."""
    index = pd.date_range("2021-03-01 00:00", periods=8, freq="15min", tz="UTC")
    df = make_frame([(10 + i, 20 + i, 5 + i, 15 + i) for i in range(8)]).set_axis(index)
    aligned = resample.align_completed(index, resample.to_timeframe(df, "H1"), "H1")

    assert aligned["h1_close"].iloc[:4].isna().all(), "la barre en cours ne doit pas fuiter"
    assert aligned["h1_close"].iloc[4] == pytest.approx(18.0)


# --------------------------------------------------------------------------------------
# Structure de prix
# --------------------------------------------------------------------------------------


def _pivot_frame() -> pd.DataFrame:
    """Sommet net en position 5, creux net en position 11."""
    highs = [10, 11, 12, 13, 14, 20, 14, 13, 12, 11, 10, 9, 12, 13, 14, 15, 16, 17]
    bars: list[Bar] = []
    for h in highs:
        bars.append((h - 0.5, h, h - 1.0, h - 0.2))
    return make_frame(bars)


def test_pivot_n_est_connu_que_W_barres_plus_tard() -> None:
    """Le decalage de confirmation est LA protection du module SMC."""
    df = _pivot_frame()
    levels = smc.swing_levels(df, width=3)
    highs = levels["swing_high"]
    # Sommet en position 5 (high = 20), confirme en position 8.
    assert highs.iloc[7] != pytest.approx(20.0), "connu trop tot = anticipation"
    assert highs.iloc[8] == pytest.approx(20.0)


def test_age_du_pivot_croit_apres_confirmation() -> None:
    df = _pivot_frame()
    levels = smc.swing_levels(df, width=3)
    assert levels["swing_high_age"].iloc[8] == pytest.approx(0.0)
    assert levels["swing_high_age"].iloc[10] == pytest.approx(2.0)


def test_bos_et_choch_sont_distingues() -> None:
    """Meme signature de prix, sens oppose : les confondre melange les regimes."""
    df = _pivot_frame()
    struct = smc.structure(df, width=3)
    both = (struct["bos"] & struct["choch"]).any()
    assert not both, "une cassure est BOS ou CHoCH, jamais les deux"
    assert struct["structure"].isin([-1, 0, 1]).all()


def test_fvg_haussier_detecte_le_saut_de_prix() -> None:
    """low[i] > high[i-2] : le marche a saute une zone en montant."""
    bars: list[Bar] = [*flat_bars(30, price=100.0, wick=0.5)]
    bars[27] = (100.0, 100.5, 99.5, 100.0)
    bars[28] = (100.0, 106.0, 100.0, 105.0)
    bars[29] = (105.0, 108.0, 104.0, 107.0)  # low 104 > high 100.5 de la barre 27
    df = make_frame(bars)
    fvg = smc.fair_value_gaps(df, min_atr=0.1)
    assert bool(fvg["fvg_bull"].iloc[29])
    assert fvg["fvg_bull_bottom"].iloc[29] == pytest.approx(100.5)
    assert fvg["fvg_bull_top"].iloc[29] == pytest.approx(104.0)


def test_micro_ecart_sous_le_seuil_n_est_pas_un_fvg() -> None:
    """Sans seuil, un instrument bruite produirait des milliers de faux FVG."""
    bars: list[Bar] = [*flat_bars(30, price=100.0, wick=0.5)]
    bars[29] = (100.0, 100.6, 100.55, 100.58)  # ecart minuscule
    df = make_frame(bars)
    assert not bool(smc.fair_value_gaps(df, min_atr=2.0)["fvg_bull"].iloc[29])


def test_balayage_et_cassure_sont_distingues() -> None:
    """Percer puis revenir = balayage. Percer et cloturer au-dela = cassure."""
    bars: list[Bar] = [*flat_bars(30, price=100.0, wick=0.5)]
    bars[28] = (100.0, 105.0, 99.5, 100.0)  # meche haute, cloture en deca -> balayage
    bars[29] = (100.0, 106.0, 99.5, 105.5)  # cloture au-dela -> cassure
    df = make_frame(bars)
    ref_high = pd.Series(101.0, index=df.index)
    ref_low = pd.Series(95.0, index=df.index)
    out = smc.liquidity_sweeps(df, ref_high, ref_low, min_wick_atr=0.0)

    assert bool(out["sweep_high"].iloc[28]) and not bool(out["breakout_high"].iloc[28])
    assert bool(out["breakout_high"].iloc[29]) and not bool(out["sweep_high"].iloc[29])


def test_premium_discount_situe_le_prix_dans_la_jambe() -> None:
    df = _pivot_frame()
    pdz = smc.premium_discount(df, width=3)
    valid = pdz["pd_position"].dropna()
    assert not valid.empty
    assert (pdz["in_discount"] & pdz["in_premium"]).sum() == 0, "les deux zones s'excluent"


def test_build_all_ne_produit_pas_de_colonnes_dupliquees() -> None:
    df = make_frame(
        [(100 + np.sin(i), 101 + np.sin(i), 99 + np.sin(i), 100.5 + np.sin(i)) for i in range(200)]
    )
    out = smc.build_all(df)
    assert out.columns.is_unique
    assert out.index.equals(df.index)
