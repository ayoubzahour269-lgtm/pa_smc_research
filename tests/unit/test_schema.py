"""Le contrat OHLCV doit echouer bruyamment, jamais silencieusement."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from tests.conftest import flat_bars, make_frame

from alphalab.data.schema import SchemaError, bar_grid_report, validate_ohlcv


def test_trame_valide_passe() -> None:
    validate_ohlcv(make_frame(flat_bars(5)), name="ok")


def test_index_naif_refuse() -> None:
    df = make_frame(flat_bars(5))
    df.index = df.index.tz_localize(None)
    with pytest.raises(SchemaError, match="tz-aware"):
        validate_ohlcv(df)


def test_index_non_utc_refuse() -> None:
    df = make_frame(flat_bars(5))
    df.index = df.index.tz_convert("Europe/Paris")
    with pytest.raises(SchemaError, match="UTC"):
        validate_ohlcv(df)


def test_index_decroissant_refuse() -> None:
    df = make_frame(flat_bars(5)).iloc[::-1]
    with pytest.raises(SchemaError, match="croissant"):
        validate_ohlcv(df)


def test_doublons_refuses() -> None:
    df = make_frame(flat_bars(3))
    df = pd.concat([df, df.iloc[[0]]]).sort_index()
    with pytest.raises(SchemaError, match="doublon"):
        validate_ohlcv(df)


def test_colonne_manquante_refusee() -> None:
    df = make_frame(flat_bars(5)).drop(columns=["low"])
    with pytest.raises(SchemaError, match="manquantes"):
        validate_ohlcv(df)


def test_nan_dans_ohlc_refuse() -> None:
    df = make_frame(flat_bars(5))
    df.loc[df.index[2], "close"] = np.nan
    with pytest.raises(SchemaError, match="NaN"):
        validate_ohlcv(df)


def test_incoherence_high_refusee() -> None:
    """Un high sous le close est une donnee impossible : elle fausserait les sorties."""
    df = make_frame(flat_bars(5))
    df.loc[df.index[1], "high"] = 50.0
    with pytest.raises(SchemaError, match="incoherences OHLC"):
        validate_ohlcv(df)


def test_incoherence_low_refusee() -> None:
    df = make_frame(flat_bars(5))
    df.loc[df.index[1], "low"] = 500.0
    with pytest.raises(SchemaError, match="incoherences OHLC"):
        validate_ohlcv(df)


def test_trame_vide_refusee() -> None:
    df = make_frame(flat_bars(3)).iloc[0:0]
    with pytest.raises(SchemaError, match="vide"):
        validate_ohlcv(df)


def test_volume_optionnel() -> None:
    df = make_frame(flat_bars(5)).drop(columns=["volume"])
    validate_ohlcv(df, require_volume=False)
    with pytest.raises(SchemaError):
        validate_ohlcv(df, require_volume=True)


def test_rapport_de_grille_compte_les_trous() -> None:
    df = make_frame(flat_bars(6))
    df = df.drop(index=df.index[3])  # un trou d'une barre
    report = bar_grid_report(df, pd.Timedelta(hours=1))
    assert report["n_bars"] == 5
    assert report["n_gaps"] == 1
