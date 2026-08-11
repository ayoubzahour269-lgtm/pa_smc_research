"""Walk-forward purge et calibration de probabilite."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphalab.model import calibration, walkforward


def _windows(n: int, *, hold: str = "6h", step: str = "1h") -> tuple[pd.Series, pd.Series]:
    entries = pd.date_range("2021-01-01", periods=n, freq=step, tz="UTC")
    exits = entries + pd.Timedelta(hold)
    return pd.Series(entries), pd.Series(exits)


# --------------------------------------------------------------------------------------
# Walk-forward
# --------------------------------------------------------------------------------------


def test_l_apprentissage_precede_toujours_le_test() -> None:
    """Un k-fold melangerait passe et futur : impossible sur une serie temporelle."""
    entries, exits = _windows(2000)
    for split in walkforward.purged_walk_forward(entries, exits, n_splits=5, min_train=10):
        assert split.train.max() < split.test.min()


def test_la_purge_retire_les_fenetres_qui_chevauchent_le_test() -> None:
    """Un trade encore ouvert au debut du test partage son evolution de prix avec lui."""
    entries, exits = _windows(2000, hold="48h")
    for split in walkforward.purged_walk_forward(
        entries, exits, n_splits=4, embargo=pd.Timedelta(0), min_train=10
    ):
        assert (pd.DatetimeIndex(exits)[split.train] < split.test_start).all()


def test_l_embargo_ecarte_la_bande_precedant_le_test() -> None:
    entries, exits = _windows(2000, hold="1h")
    embargo = pd.Timedelta(days=2)
    splits = list(
        walkforward.purged_walk_forward(entries, exits, n_splits=4, embargo=embargo, min_train=10)
    )
    assert splits
    for split in splits:
        latest = pd.DatetimeIndex(exits)[split.train].max()
        assert latest < split.test_start - embargo


def test_un_embargo_plus_large_reduit_l_apprentissage() -> None:
    entries, exits = _windows(2000)
    petit = list(
        walkforward.purged_walk_forward(entries, exits, embargo=pd.Timedelta(0), min_train=10)
    )
    grand = list(
        walkforward.purged_walk_forward(entries, exits, embargo=pd.Timedelta(days=10), min_train=10)
    )
    assert sum(s.n_train for s in grand) < sum(s.n_train for s in petit)


def test_donnees_non_triees_refusees() -> None:
    entries, exits = _windows(500)
    with pytest.raises(ValueError, match="triees"):
        list(walkforward.purged_walk_forward(entries[::-1].reset_index(drop=True), exits))


def test_horodatages_identiques_acceptes() -> None:
    """Plusieurs familles peuvent se declencher sur la MEME barre : ce n'est pas une erreur."""
    stamps = ["2021-01-01 00:00"] * 3 + ["2021-01-01 01:00"] * 997
    entries = pd.Series(pd.DatetimeIndex(stamps, tz="UTC"))
    exits = entries + pd.Timedelta("2h")
    assert list(walkforward.purged_walk_forward(entries, exits, min_train=10)) is not None


def test_couverture_rapportee() -> None:
    entries, exits = _windows(1000)
    splits = list(walkforward.purged_walk_forward(entries, exits, min_train=10))
    info = walkforward.coverage(splits, 1000)
    assert info["plis"] == len(splits)
    assert 0.0 < info["couverture_test"] <= 1.0


# --------------------------------------------------------------------------------------
# Metriques de calibration
# --------------------------------------------------------------------------------------


def test_brier_recompense_la_certitude_juste() -> None:
    y = np.array([1, 1, 0, 0])
    assert calibration.brier(y, np.array([1.0, 1.0, 0.0, 0.0])) == pytest.approx(0.0)
    assert calibration.brier(y, np.array([0.0, 0.0, 1.0, 1.0])) == pytest.approx(1.0)


def test_ece_nul_pour_un_modele_parfaitement_calibre() -> None:
    """Annoncer 30 % et voir 30 % de gains : erreur de calibration nulle."""
    rng = np.random.default_rng(0)
    p = np.full(1000, 0.3)
    y = (rng.uniform(size=1000) < 0.3).astype(int)
    assert calibration.expected_calibration_error(y, p) < 0.05


def test_ece_eleve_pour_un_modele_surconfiant() -> None:
    p = np.full(1000, 0.9)
    y = np.zeros(1000, dtype=int)
    assert calibration.expected_calibration_error(y, p) > 0.8


def test_table_de_fiabilite_confronte_annonce_et_observe() -> None:
    p = np.concatenate([np.full(500, 0.2), np.full(500, 0.8)])
    y = np.concatenate([np.zeros(500, dtype=int), np.ones(500, dtype=int)])
    table = calibration.reliability_table(y, p)
    assert len(table) == 2
    assert set(table.columns) == {"tranche", "n", "annonce", "observe", "ecart"}


# --------------------------------------------------------------------------------------
# Ajustement
# --------------------------------------------------------------------------------------


def _learnable(n: int = 3000, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray, list]:
    """Jeu ou une feature porte reellement de l'information."""
    rng = np.random.default_rng(seed)
    signal = rng.normal(size=n)
    noise = rng.normal(size=n, scale=3.0)
    proba = 1.0 / (1.0 + np.exp(-1.5 * signal))
    y = (rng.uniform(size=n) < proba).astype(np.int64)
    X = pd.DataFrame({"signal": signal, "bruit": noise})
    entries, exits = _windows(n, hold="30min")
    splits = list(walkforward.purged_walk_forward(entries, exits, n_splits=5, min_train=100))
    return X, y, splits


def test_un_signal_reel_est_detecte_comme_utile() -> None:
    """Controle de sensibilite : sans lui, un module casse qui rejette tout semblerait rigoureux."""
    X, y, splits = _learnable()
    model = calibration.fit(X, y, splits)
    assert model is not None
    assert model.skill > 0.05
    assert model.is_useful


def test_du_bruit_pur_n_est_pas_declare_utile() -> None:
    """Le point essentiel : un modele sans information doit le DIRE."""
    rng = np.random.default_rng(1)
    n = 3000
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = (rng.uniform(size=n) < 0.35).astype(np.int64)
    entries, exits = _windows(n, hold="30min")
    splits = list(walkforward.purged_walk_forward(entries, exits, n_splits=5, min_train=100))
    model = calibration.fit(X, y, splits)
    assert model is not None
    assert not model.is_useful, f"skill = {model.skill}"


def test_l_ece_n_est_pas_auto_evalue() -> None:
    """L'isotonique doit etre ajustee et evaluee sur des points DIFFERENTS.

    Sinon la courbe passe par les points qui ont servi a la tracer, l'ECE tombe a zero
    et le chiffre publie est rassurant mais faux.
    """
    X, y, splits = _learnable()
    model = calibration.fit(X, y, splits)
    assert model is not None
    assert model.diagnostics["n_evaluation"] < model.diagnostics["n_hors_pli"]
    assert "1re moitie" in model.diagnostics["note_calibration"]


def test_sans_pli_exploitable_le_modele_est_absent() -> None:
    X = pd.DataFrame({"a": [1.0, 2.0]})
    assert calibration.fit(X, np.array([0, 1]), []) is None


def test_esperance_r_depuis_la_probabilite() -> None:
    """E[R] = p x R_objectif - (1 - p) : c'est ce nombre qui pilote le grade."""
    assert calibration.expected_r(0.5, 2.0) == pytest.approx(0.5)
    assert calibration.expected_r(1.0 / 3.0, 2.0) == pytest.approx(0.0, abs=1e-9)
    assert calibration.expected_r(0.25, 2.0) == pytest.approx(-0.25)
    assert calibration.expected_r(0.5, 2.0, cost_r=0.1) == pytest.approx(0.4)
