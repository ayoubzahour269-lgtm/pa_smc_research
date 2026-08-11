"""Toute feature du paquet doit etre causale. Aucune exception.

Ce fichier est la garde principale du depot : si l'un de ces tests echoue, une feature
utilise de l'information future et tout resultat de backtest qui en depend est faux.
"""

from __future__ import annotations

import pandas as pd
import pytest

from alphalab.features import indicators, resample, sessions, smc
from alphalab.features.causality import assert_causal, check_causal, synthetic_ohlcv

DF = synthetic_ohlcv(2500, seed=7)


# --------------------------------------------------------------------------------------
# Le verificateur lui-meme doit fonctionner
# --------------------------------------------------------------------------------------


def test_le_verificateur_detecte_une_anticipation_evidente() -> None:
    """Sans ce test, un verificateur casse validerait tout le reste en silence."""
    violations = check_causal(lambda d: d["close"].shift(-1).rename("futur"), DF, n_probes=5)
    assert violations
    assert "information future" in violations[0]


def test_le_verificateur_accepte_une_feature_causale() -> None:
    assert check_causal(lambda d: d["close"].shift(1).rename("passe"), DF, n_probes=5) == []


def test_le_verificateur_detecte_une_moyenne_centree() -> None:
    """Piege classique : une fenetre `center=True` regarde la moitie de son span devant."""
    violations = check_causal(
        lambda d: d["close"].rolling(11, center=True).mean().rename("centree"), DF, n_probes=5
    )
    assert violations


def test_le_verificateur_detecte_une_normalisation_globale() -> None:
    """Normaliser par une statistique de TOUT l'historique fait fuiter le futur.

    Piege tres frequent en apprentissage automatique : standardiser les features sur
    l'ensemble du jeu avant de decouper apprentissage et test.
    """
    violations = check_causal(
        lambda d: (d["close"] / d["close"].mean()).rename("norm"), DF, n_probes=5
    )
    assert violations


def test_historique_trop_court_leve() -> None:
    with pytest.raises(ValueError, match="trop court"):
        check_causal(lambda d: d["close"], DF.iloc[:50])


# --------------------------------------------------------------------------------------
# Indicateurs
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("nom", "fn"),
    [
        ("true_range", indicators.true_range),
        ("atr_wilder", lambda d: indicators.atr(d, 14)),
        ("atr_sma", lambda d: indicators.atr(d, 14, wilder=False)),
        ("ema", lambda d: indicators.ema(d["close"], 50)),
        ("sma", lambda d: indicators.sma(d["close"], 20)),
        ("rsi", lambda d: indicators.rsi(d["close"], 14)),
        ("adx", lambda d: indicators.adx(d, 14)),
        ("rolling_high", lambda d: indicators.rolling_high(d, 20)),
        ("rolling_low", lambda d: indicators.rolling_low(d, 20)),
        ("realized_volatility", lambda d: indicators.realized_volatility(d, 20)),
        ("atr_ratio", lambda d: indicators.atr_ratio(d)),
        ("body_and_wicks", indicators.body_and_wicks),
        ("zscore", lambda d: indicators.zscore(d["close"], 30)),
    ],
)
def test_indicateur_causal(nom: str, fn) -> None:
    assert_causal(fn, DF, label=nom)


def test_vwap_ancre_causal() -> None:
    assert_causal(
        lambda d: indicators.anchored_vwap(d, sessions.trading_day(pd.DatetimeIndex(d.index))),
        DF,
        label="anchored_vwap",
    )


# --------------------------------------------------------------------------------------
# Seances
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("nom", "fn"),
    [
        ("session_label", lambda d: sessions.session_label(pd.DatetimeIndex(d.index)).to_frame()),
        ("session_masks", lambda d: sessions.session_masks(pd.DatetimeIndex(d.index))),
        ("trading_day", lambda d: sessions.trading_day(pd.DatetimeIndex(d.index)).to_frame()),
        ("session_progress", sessions.session_progress),
        ("previous_day_levels", sessions.previous_day_levels),
    ],
)
def test_seance_causale(nom: str, fn) -> None:
    assert_causal(fn, DF, label=nom)


@pytest.mark.parametrize("window", [*sessions.SESSIONS, *sessions.KILLZONES])
def test_range_de_seance_causal(window: sessions.SessionWindow) -> None:
    """Le range d'une seance ne doit exister qu'une fois la seance terminee."""
    assert_causal(
        lambda d, w=window: sessions.completed_window_range(d, w),
        DF,
        label=f"completed_window_range({window.name})",
    )


def test_range_asiatique_indisponible_pendant_la_seance() -> None:
    """Controle explicite du piege : a 03h, le range asiatique du jour n'existe pas encore."""
    asia = next(w for w in sessions.SESSIONS if w.name == "asie")
    out = sessions.completed_window_range(DF, asia)
    index = pd.DatetimeIndex(DF.index)
    day = sessions.trading_day(index)

    # Premiere journee complete, a 03h UTC : la valeur doit provenir d'une seance
    # ANTERIEURE, jamais de la seance en cours.
    mid_session = out[(index.hour == 3) & (day == day.iloc[-1])]
    if not mid_session.empty:
        same_day_high = DF.loc[(index.hour < 3) & (day == day.iloc[-1]), "high"]
        if not same_day_high.empty:
            assert not (mid_session["asie_high"] == same_day_high.max()).all()


# --------------------------------------------------------------------------------------
# Structure de prix (SMC)
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("nom", "fn"),
    [
        ("swing_levels", lambda d: smc.swing_levels(d, 3)),
        ("swing_levels_w5", lambda d: smc.swing_levels(d, 5)),
        ("structure", lambda d: smc.structure(d, 3)),
        ("premium_discount", lambda d: smc.premium_discount(d, 3)),
        ("fair_value_gaps", smc.fair_value_gaps),
        ("order_blocks", lambda d: smc.order_blocks(d, width=3)),
        ("equal_levels", lambda d: smc.equal_levels(d)),
        ("build_all", smc.build_all),
    ],
)
def test_smc_causal(nom: str, fn) -> None:
    assert_causal(fn, DF, label=nom)


# --------------------------------------------------------------------------------------
# Multi-timeframe
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("timeframe", ["H1", "H4", "D1"])
def test_contexte_superieur_causal(timeframe: str) -> None:
    """La barre superieure en cours ne doit jamais fuiter vers le timeframe fin.

    C'est le second piege du multi-timeframe, apres celui des pivots : la barre H4 qui
    couvre 12h-16h n'est connue qu'a 16h.
    """

    def fn(d: pd.DataFrame) -> pd.DataFrame:
        return resample.align_completed(
            pd.DatetimeIndex(d.index), resample.to_timeframe(d, timeframe), timeframe
        )

    assert_causal(fn, DF, label=f"align_completed({timeframe})")


def test_contexte_multi_timeframe_causal() -> None:
    assert_causal(resample.multi_timeframe_context, DF, label="multi_timeframe_context")


# --------------------------------------------------------------------------------------
# Liquidite
# --------------------------------------------------------------------------------------


def test_sweeps_causaux() -> None:
    def fn(d: pd.DataFrame) -> pd.DataFrame:
        levels = smc.equal_levels(d)
        return smc.liquidity_sweeps(d, levels["eq_high"], levels["eq_low"])

    assert_causal(fn, DF, label="liquidity_sweeps")


def test_la_detection_brute_de_pivots_est_bien_non_causale() -> None:
    """Documente pourquoi `_raw_pivots` reste prive.

    Si ce test se met a passer sans violation, c'est que la detection brute a change de
    semantique — et que `swing_levels` protege peut-etre desormais dans le vide.
    """
    violations = check_causal(
        lambda d: smc._raw_pivots(d, 3)[0].rename("pivot_brut").to_frame(), DF, n_probes=10
    )
    assert violations, "la detection brute devrait anticiper de W barres"
