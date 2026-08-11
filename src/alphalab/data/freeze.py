"""Acquisition et gel de donnees Dukascopy.

IMPORTANT — ce module ne fonctionne PAS dans l'environnement de developpement de ce
projet : l'acces reseau aux fournisseurs de donnees de marche y est bloque. Il est
concu pour etre execute sur votre machine, ou Dukascopy est joignable :

    pip install -e ".[fetch]"
    alphalab freeze EURUSD --tf M15 --side BID,ASK

Les snapshots produits sont ensuite committes. Le reste de la plateforme les detecte
automatiquement : aucun code a modifier pour activer un nouveau symbole.
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any

import pandas as pd

from alphalab.config import BAR_DURATION, UNIVERSE
from alphalab.data import registry, snapshot
from alphalab.data.schema import bar_grid_report, validate_ohlcv
from alphalab.data.snapshot import SnapshotRef

_MAX_RETRIES = 4
_BACKOFF_SECONDS = (2, 4, 8, 16)


class FetchError(RuntimeError):
    """Le telechargement a echoue ou la dependance optionnelle est absente."""


def _require_dukascopy() -> Any:
    try:
        import dukascopy_python
    except ImportError as exc:  # pragma: no cover - depend de l'environnement
        raise FetchError(
            "dukascopy-python n'est pas installe. Installez les extras de "
            'telechargement : pip install -e ".[fetch]"'
        ) from exc
    return dukascopy_python


def _resolve(dk: Any, timeframe: str, side: str, instrument_const: str) -> tuple[Any, Any, Any]:
    from dukascopy_python import instruments

    intervals = {
        "M1": "INTERVAL_MIN_1",
        "M5": "INTERVAL_MIN_5",
        "M15": "INTERVAL_MIN_15",
        "H1": "INTERVAL_HOUR_1",
        "H4": "INTERVAL_HOUR_4",
        "D1": "INTERVAL_DAY_1",
    }
    if timeframe not in intervals:
        raise FetchError(f"Timeframe non gere : {timeframe}")
    if side not in ("BID", "ASK"):
        raise FetchError(f"Cote non gere : {side}")
    if not hasattr(instruments, instrument_const):
        raise FetchError(f"Instrument inconnu de dukascopy-python : {instrument_const}")
    return (
        getattr(instruments, instrument_const),
        getattr(dk, intervals[timeframe]),
        getattr(dk, f"OFFER_SIDE_{side}"),
    )


def fetch(
    symbol: str, timeframe: str, side: str, start: datetime, end: datetime
) -> pd.DataFrame:
    """Telecharge un intervalle, annee par annee, avec reprise en cas d'echec reseau.

    Le decoupage annuel n'est pas cosmetique : sur les petits timeframes une requete
    unique sur dix ans depasse largement ce que le fournisseur accepte de servir.
    """
    dk = _require_dukascopy()
    spec = UNIVERSE.get(symbol)
    if spec is None or not spec.dukascopy_id:
        raise FetchError(
            f"{symbol} n'a pas d'identifiant Dukascopy dans alphalab.config.UNIVERSE. "
            "Ajoutez-le avant de telecharger."
        )
    instrument, interval, offer = _resolve(dk, timeframe, side, spec.dukascopy_id)

    frames: list[pd.DataFrame] = []
    for year in range(start.year, end.year + 1):
        chunk_start = max(datetime(year, 1, 1), start)
        chunk_end = min(datetime(year + 1, 1, 1), end)
        if chunk_start >= chunk_end:
            continue
        last_error: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                frames.append(dk.fetch(instrument, interval, offer, chunk_start, chunk_end))
                last_error = None
                break
            except Exception as exc:  # pragma: no cover - depend du reseau
                last_error = exc
                if attempt < _MAX_RETRIES - 1:
                    time.sleep(_BACKOFF_SECONDS[attempt])
        if last_error is not None:  # pragma: no cover - depend du reseau
            raise FetchError(
                f"Echec du telechargement {symbol} {timeframe} {side} {year}"
            ) from last_error

    if not frames:
        raise FetchError(f"Aucune donnee renvoyee pour {symbol} {timeframe} {side}")

    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    index = pd.DatetimeIndex(df.index)
    df.index = index.tz_localize("UTC") if index.tz is None else index.tz_convert("UTC")
    df.index.name = "timestamp"
    return df


def freeze(
    symbol: str,
    timeframe: str,
    side: str,
    start: datetime,
    end: datetime,
    *,
    version: str = "v1",
    note: str | None = None,
) -> dict[str, Any]:
    """Telecharge puis gele un snapshot immuable. Ne reecrit jamais un fichier existant."""
    ref = SnapshotRef(
        symbol=symbol,
        timeframe=timeframe,
        side=side,
        start=start.strftime("%Y-%m-%d"),
        end=(end - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        version=version,
    )
    if ref.exists():
        return {"skipped": True, "basename": ref.basename, "reason": "deja gele (immuable)"}

    df = fetch(symbol, timeframe, side, start, end)
    validate_ohlcv(df, name=ref.basename)

    spec = UNIVERSE[symbol]
    grid = bar_grid_report(df, BAR_DURATION[timeframe])
    # Mediane de barres par jour et par annee : detecte un changement de regime de
    # seance du fournisseur, qui invaliderait toute comparaison longue.
    index = pd.DatetimeIndex(df.index)
    per_year = {
        int(year): int(group.groupby(pd.DatetimeIndex(group.index).date).size().median())
        for year, group in df.groupby(index.year)
    }
    extra: dict[str, Any] = {
        "instrument_label": spec.label,
        "instrument_constant": spec.dukascopy_id,
        "source": "Dukascopy Bank SA (via dukascopy-python)",
        "grille": grid,
        "barres_par_jour_mediane_par_annee": per_year,
    }
    if note:
        extra["session_note"] = note

    manifest = snapshot.write(ref, df, extra_manifest=extra)
    registry.refresh()
    return {"skipped": False, "basename": ref.basename, **manifest}
