"""Registre et couche de cout, sur des snapshots synthetiques.

Ces tests verifient le comportement qui compte le plus en pratique : deposer un
snapshot suffit a activer un symbole, et l'absence du cote ASK doit produire une
erreur nette plutot qu'un cout invente.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from tests.conftest import flat_bars, make_frame

from alphalab.data import costs, registry, snapshot
from alphalab.data.costs import CostUnavailableError
from alphalab.data.snapshot import SnapshotError, SnapshotRef

START, END = "2020-01-01", "2020-12-31"


@pytest.fixture
def snap_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    monkeypatch.setattr(snapshot, "SNAPSHOT_DIR", tmp_path)
    monkeypatch.setattr(registry, "SNAPSHOT_DIR", tmp_path)
    snapshot.clear_cache()
    registry.refresh()
    yield tmp_path
    snapshot.clear_cache()
    registry.refresh()


def _put(
    symbol: str, tf: str, side: str, *, offset: float = 0.0, version: str = "v1", n: int = 48
) -> SnapshotRef:
    ref = SnapshotRef(symbol, tf, side, START, END, version)
    df = make_frame(flat_bars(n))
    for col in ("open", "high", "low", "close"):
        df[col] = df[col] + offset
    snapshot.write(ref, df)
    registry.refresh()
    return ref


def test_depot_d_un_snapshot_active_le_symbole(snap_dir: Path) -> None:
    assert registry.available_symbols() == []
    _put("EURUSD", "M15", "BID")
    assert registry.available_symbols() == ["EURUSD"]
    assert registry.available_timeframes("EURUSD") == ["M15"]


def test_symbole_tradable_exige_les_deux_cotes(snap_dir: Path) -> None:
    _put("EURUSD", "M15", "BID")
    assert registry.tradable("M15") == []
    _put("EURUSD", "M15", "ASK", offset=0.0002)
    assert registry.tradable("M15") == ["EURUSD"]


def test_la_version_la_plus_recente_gagne(snap_dir: Path) -> None:
    _put("XAUUSD", "H1", "BID", version="v1")
    _put("XAUUSD", "H1", "BID", version="v2")
    ref = registry.find("XAUUSD", "H1", "BID")
    assert ref is not None and ref.version == "v2"


def test_require_donne_une_erreur_actionnable(snap_dir: Path) -> None:
    with pytest.raises(SnapshotError, match="alphalab freeze"):
        registry.require("EURUSD", "M15", "BID")


def test_fichier_etranger_est_ignore(snap_dir: Path) -> None:
    (snap_dir / "notes.manifest.json").write_text("{}", encoding="utf-8")
    registry.refresh()
    assert registry.available_symbols() == []


def test_spread_reel_est_la_difference_des_ouvertures(snap_dir: Path) -> None:
    _put("EURUSD", "M15", "BID")
    _put("EURUSD", "M15", "ASK", offset=0.5)
    spread = costs.real_spread("EURUSD", "M15")
    assert len(spread) == 48
    assert spread.min() == pytest.approx(0.5)
    assert spread.max() == pytest.approx(0.5)


def test_ask_manquant_leve_plutot_que_d_inventer_un_cout(snap_dir: Path) -> None:
    """Le projet a deja perdu une session a cause d'un cout suppose. Plus jamais."""
    _put("EURUSD", "M15", "BID")
    with pytest.raises(CostUnavailableError, match="ASK"):
        costs.cost_model("EURUSD", "M15")


def test_majoration_du_modele_de_cout(snap_dir: Path) -> None:
    _put("EURUSD", "M15", "BID")
    _put("EURUSD", "M15", "ASK", offset=0.4)
    model = costs.cost_model("EURUSD", "M15")
    assert model.describe()["mediane"] == pytest.approx(0.4)
    assert model.stressed(1.5).describe()["mediane"] == pytest.approx(0.6)


def test_profil_horaire_du_spread(snap_dir: Path) -> None:
    _put("EURUSD", "H1", "BID")
    _put("EURUSD", "H1", "ASK", offset=0.2)
    profile = costs.hourly_profile(costs.real_spread("EURUSD", "H1"))
    assert profile.index.name == "heure_utc"
    assert len(profile) == 24
    assert profile["mediane"].max() == pytest.approx(0.2)
    assert profile["x_mediane_globale"].max() == pytest.approx(1.0)


def test_alignement_du_cout_sur_un_index_de_travail(snap_dir: Path) -> None:
    """Une barre hors grille doit ressortir en NaN, pas en zero silencieux."""
    _put("EURUSD", "H1", "BID")
    _put("EURUSD", "H1", "ASK", offset=0.3)
    model = costs.cost_model("EURUSD", "H1")
    index = model.spread.index[:5].append(
        make_frame(flat_bars(1), start="2035-01-01 00:00").index
    )
    aligned = model.aligned_to(index)  # type: ignore[arg-type]
    assert aligned[:5].tolist() == pytest.approx([0.3] * 5)
    assert aligned[-1] != aligned[-1]  # NaN
