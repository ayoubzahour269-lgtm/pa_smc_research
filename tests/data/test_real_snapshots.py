"""Integrite des snapshots reels versionnes dans le depot.

Marques `slow` : ces tests lisent ~86 Mo et calculent des sha256. Ils sont la garantie
que la donnee sur laquelle reposent tous les resultats publies n'a pas bouge.
"""

from __future__ import annotations

import pytest

from alphalab.config import BAR_DURATION, HOLDOUT_START
from alphalab.data import costs, registry, snapshot

pytestmark = pytest.mark.slow

EXPECTED_BARS = {
    ("XAUUSD", "H1"): 68_041,
    ("XAUUSD", "M15"): 271_660,
}


def test_le_depot_contient_les_snapshots_attendus() -> None:
    symbols = registry.available_symbols()
    assert "XAUUSD" in symbols
    assert "NAS100" in symbols


@pytest.mark.parametrize("symbol", ["XAUUSD", "NAS100"])
@pytest.mark.parametrize("timeframe", ["H1", "M15"])
@pytest.mark.parametrize("side", ["BID", "ASK"])
def test_sha256_conforme_au_manifeste(symbol: str, timeframe: str, side: str) -> None:
    ref = registry.require(symbol, timeframe, side)
    assert snapshot.verify(ref) == snapshot.read_manifest(ref)["sha256"]


@pytest.mark.parametrize("symbol", ["XAUUSD", "NAS100"])
@pytest.mark.parametrize("timeframe", ["H1", "M15"])
def test_contrat_ohlcv_respecte(symbol: str, timeframe: str) -> None:
    """`snapshot.load` valide le contrat ; l'appel suffit a le prouver."""
    df = snapshot.load(registry.require(symbol, timeframe, "BID"), copy=False)
    assert len(df) > 10_000
    assert str(df.index.tz) == "UTC"


@pytest.mark.parametrize(
    ("symbol", "timeframe", "n"), [(s, t, n) for (s, t), n in EXPECTED_BARS.items()]
)
def test_nombre_de_barres_fige(symbol: str, timeframe: str, n: int) -> None:
    """Verrou sur le compte exact : une donnee qui grandit n'est plus la meme donnee."""
    assert len(snapshot.load(registry.require(symbol, timeframe, "BID"), copy=False)) == n


@pytest.mark.parametrize("symbol", ["XAUUSD", "NAS100"])
@pytest.mark.parametrize("timeframe", ["H1", "M15"])
def test_grilles_bid_et_ask_identiques(symbol: str, timeframe: str) -> None:
    """Sans grilles alignees, le spread mesure comparerait deux instants differents."""
    bid = snapshot.load(registry.require(symbol, timeframe, "BID"), copy=False)
    ask = snapshot.load(registry.require(symbol, timeframe, "ASK"), copy=False)
    assert bid.index.equals(ask.index)


@pytest.mark.parametrize("symbol", ["XAUUSD", "NAS100"])
@pytest.mark.parametrize("timeframe", ["H1", "M15"])
def test_spread_reel_positif_et_sans_trou(symbol: str, timeframe: str) -> None:
    """Un spread negatif signalerait une inversion BID/ASK — donnee inutilisable."""
    spread = costs.real_spread(symbol, timeframe)
    assert int(spread.isna().sum()) == 0
    assert int((spread < 0).sum()) == 0
    assert float(spread.median()) > 0


@pytest.mark.parametrize("symbol", ["XAUUSD", "NAS100"])
def test_le_holdout_est_present_mais_reste_hors_analyse(symbol: str) -> None:
    """Le hold-out doit exister dans la donnee, et n'etre ouvert qu'une fois, a la fin."""
    df = snapshot.load(registry.require(symbol, "H1", "BID"), copy=False)
    assert (df.index >= HOLDOUT_START).any()
    assert (df.index < HOLDOUT_START).any()


@pytest.mark.parametrize("timeframe", ["H1", "M15"])
def test_les_deux_symboles_sont_tradables(timeframe: str) -> None:
    assert set(registry.tradable(timeframe)) >= {"XAUUSD", "NAS100"}


def test_regime_de_seance_du_nasdaq_documente() -> None:
    """Le Nasdaq Dukascopy change de regime de seance avant 2019.

    Ce test fige la raison pour laquelle `IN_SAMPLE_START["NAS100"]` vaut 2019 :
    avant, le flux compte nettement moins de barres par jour, et melanger les deux
    reviendrait a etudier deux marches differents sous un meme nom.
    """
    df = snapshot.load(registry.require("NAS100", "H1", "BID"), copy=False)
    avant = df[(df.index >= "2016-01-01") & (df.index < "2017-01-01")]
    apres = df[(df.index >= "2020-01-01") & (df.index < "2021-01-01")]
    bars_avant = len(avant) / avant.index.normalize().nunique()
    bars_apres = len(apres) / apres.index.normalize().nunique()
    assert bars_avant < bars_apres * 0.8, (bars_avant, bars_apres)
    assert BAR_DURATION["H1"].total_seconds() == 3600
