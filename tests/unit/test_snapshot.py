"""Immuabilite des snapshots : nommage, verification sha256, refus d'ecrasement."""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.conftest import flat_bars, make_frame

from alphalab.data import snapshot
from alphalab.data.snapshot import SnapshotError, SnapshotRef


@pytest.fixture
def snap_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(snapshot, "SNAPSHOT_DIR", tmp_path)
    snapshot.clear_cache()
    yield tmp_path
    snapshot.clear_cache()


REF = SnapshotRef("XAUUSD", "H1", "BID", "2015-01-01", "2026-06-30", "v1")


def test_nom_de_fichier_est_deterministe() -> None:
    assert REF.basename == "XAUUSD_H1_BID_2015-01-01_2026-06-30_v1"
    assert REF.csv_path.name.endswith(".csv")


def test_analyse_du_nom_est_reciproque() -> None:
    assert SnapshotRef.from_basename(REF.basename) == REF


def test_analyse_supporte_un_symbole_avec_underscore() -> None:
    ref = SnapshotRef.from_basename("US_TECH_M15_ASK_2019-01-01_2024-12-31_v2")
    assert ref.symbol == "US_TECH"
    assert ref.timeframe == "M15"
    assert ref.side == "ASK"
    assert ref.version == "v2"


def test_nom_non_conforme_leve() -> None:
    with pytest.raises(SnapshotError, match="non conforme"):
        SnapshotRef.from_basename("nimportequoi")


def test_ecriture_puis_lecture_conserve_les_donnees(snap_dir: Path) -> None:
    df = make_frame(flat_bars(24))
    manifest = snapshot.write(REF, df)
    assert manifest["sha256"]
    assert manifest["symbol"] == "XAUUSD"
    loaded = snapshot.load(REF)
    assert len(loaded) == 24
    assert loaded.index.equals(df.index)
    assert loaded["close"].tolist() == df["close"].tolist()


def test_reecriture_refusee(snap_dir: Path) -> None:
    """Le refus d'ecrasement est la garantie de reproductibilite du projet."""
    df = make_frame(flat_bars(10))
    snapshot.write(REF, df)
    with pytest.raises(SnapshotError, match="immuable"):
        snapshot.write(REF, df)


def test_csv_modifie_est_detecte(snap_dir: Path) -> None:
    """Toucher au CSV apres coup doit provoquer un echec net, pas un resultat different."""
    snapshot.write(REF, make_frame(flat_bars(10)))
    snapshot.clear_cache()
    with REF.csv_path.open("a", encoding="utf-8") as fh:
        fh.write("2030-01-01 00:00:00+00:00,1,1,1,1,1\n")
    with pytest.raises(SnapshotError, match="sha256 different"):
        snapshot.load(REF)


def test_snapshot_absent_leve(snap_dir: Path) -> None:
    with pytest.raises(SnapshotError, match="Snapshot absent"):
        snapshot.load(REF)


def test_manifeste_absent_leve(snap_dir: Path) -> None:
    snapshot.write(REF, make_frame(flat_bars(5)))
    REF.manifest_path.unlink()
    snapshot.clear_cache()
    with pytest.raises(SnapshotError, match="Manifeste absent"):
        snapshot.load(REF)


def test_copie_par_defaut_protege_le_cache(snap_dir: Path) -> None:
    """Modifier la trame rendue ne doit pas polluer les lectures suivantes."""
    snapshot.write(REF, make_frame(flat_bars(10)))
    first = snapshot.load(REF)
    first["close"] = -1.0
    second = snapshot.load(REF)
    assert (second["close"] == 100.0).all()
