"""Snapshots immuables : nommage, lecture verifiee, ecriture protegee.

Regle du projet, heritee et conservee : un snapshot ne se reecrit JAMAIS. Toute
nouvelle donnee est une nouvelle version. C'est ce qui permet de comparer deux
resultats a des mois d'intervalle et de savoir que la difference vient du code, pas
d'une donnee qui a bouge sous les pieds.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from alphalab.config import SNAPSHOT_DIR
from alphalab.data.schema import bar_grid_report, validate_ohlcv

_CHUNK = 1 << 20


class SnapshotError(RuntimeError):
    """Snapshot absent, corrompu, ou en desaccord avec son manifeste."""


@dataclass(frozen=True, slots=True)
class SnapshotRef:
    """Identite d'un snapshot. Le nom de fichier en decoule entierement."""

    symbol: str
    timeframe: str
    side: str
    start: str
    end: str
    version: str = "v1"

    @property
    def basename(self) -> str:
        return f"{self.symbol}_{self.timeframe}_{self.side}_{self.start}_{self.end}_{self.version}"

    @property
    def csv_path(self) -> Path:
        return SNAPSHOT_DIR / f"{self.basename}.csv"

    @property
    def manifest_path(self) -> Path:
        return SNAPSHOT_DIR / f"{self.basename}.manifest.json"

    def exists(self) -> bool:
        return self.csv_path.is_file() and self.manifest_path.is_file()

    @classmethod
    def from_basename(cls, basename: str) -> SnapshotRef:
        """Analyse `SYMBOLE_TF_SIDE_debut_fin_version`.

        Le symbole peut contenir des underscores ; on decoupe donc par la droite.
        """
        parts = basename.rsplit("_", 5)
        if len(parts) != 6:
            raise SnapshotError(f"Nom de snapshot non conforme : {basename!r}")
        symbol, timeframe, side, start, end, version = parts
        return cls(symbol, timeframe, side, start, end, version)

    def __str__(self) -> str:  # pragma: no cover - confort de debogage
        return self.basename


def sha256_file(path: Path) -> str:
    """sha256 d'un fichier, lu par blocs (les snapshots pesent des dizaines de Mo)."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(ref: SnapshotRef) -> dict[str, Any]:
    if not ref.manifest_path.is_file():
        raise SnapshotError(f"Manifeste absent : {ref.manifest_path}")
    data: dict[str, Any] = json.loads(ref.manifest_path.read_text(encoding="utf-8"))
    return data


def verify(ref: SnapshotRef) -> str:
    """Verifie le sha256 du CSV contre son manifeste et renvoie l'empreinte."""
    if not ref.csv_path.is_file():
        raise SnapshotError(f"Snapshot absent : {ref.csv_path}")
    manifest = read_manifest(ref)
    expected = manifest.get("sha256")
    actual = sha256_file(ref.csv_path)
    if expected != actual:
        raise SnapshotError(
            f"sha256 different pour {ref.basename} : manifeste={expected} fichier={actual}. "
            "Le snapshot a ete modifie — il est immuable par contrat."
        )
    return actual


@lru_cache(maxsize=32)
def _read_verified(basename: str, do_verify: bool) -> pd.DataFrame:
    ref = SnapshotRef.from_basename(basename)
    if do_verify:
        verify(ref)
    elif not ref.csv_path.is_file():
        raise SnapshotError(f"Snapshot absent : {ref.csv_path}")
    df = pd.read_csv(ref.csv_path, index_col=0, parse_dates=True)
    index = pd.DatetimeIndex(df.index)
    df.index = index.tz_localize("UTC") if index.tz is None else index.tz_convert("UTC")
    df.index.name = "timestamp"
    validate_ohlcv(df, name=basename)
    return df


def load(ref: SnapshotRef, *, do_verify: bool = True, copy: bool = True) -> pd.DataFrame:
    """Charge un snapshot valide et verifie.

    Le resultat est mis en cache. `copy=True` (defaut) rend une copie : les appelants
    peuvent ajouter des colonnes sans corrompre le cache. Passer `copy=False`
    uniquement dans du code qui ne modifie rien.
    """
    df = _read_verified(ref.basename, do_verify)
    return df.copy() if copy else df


def clear_cache() -> None:
    """Vide le cache de lecture (utile en test)."""
    _read_verified.cache_clear()


def write(
    ref: SnapshotRef,
    df: pd.DataFrame,
    *,
    extra_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Ecrit un snapshot et son manifeste. Refuse d'ecraser un fichier existant.

    Cette fonction est la seule voie d'entree de donnees dans le depot. Le refus
    d'ecrasement n'est pas une precaution cosmetique : c'est ce qui garantit qu'un
    resultat publie reste reproductible.
    """
    if ref.csv_path.exists() or ref.manifest_path.exists():
        raise SnapshotError(
            f"{ref.basename} existe deja. Un snapshot est immuable : creez une nouvelle "
            f"version (v2, v3, ...) plutot que de reecrire celle-ci."
        )
    validate_ohlcv(df, name=ref.basename)
    ref.csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ref.csv_path)

    manifest: dict[str, Any] = {
        "symbol": ref.symbol,
        "timeframe": ref.timeframe,
        "offer_side": ref.side,
        "version": ref.version,
        "timezone": "UTC",
        "start": str(df.index[0]),
        "end": str(df.index[-1]),
        "columns": list(df.columns),
        "close_min": float(df["close"].min()),
        "close_max": float(df["close"].max()),
        "written_at_utc": datetime.now(UTC).isoformat(),
        "csv_file": ref.csv_path.name,
        "sha256": sha256_file(ref.csv_path),
        "note": "IMMUABLE — ne jamais reecrire. Toute nouvelle donnee = nouvelle version.",
    }
    if extra_manifest:
        manifest.update(extra_manifest)
    ref.manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest


def grid_report(ref: SnapshotRef, expected: pd.Timedelta) -> dict[str, object]:
    """Rapport de regularite de grille pour un snapshot (diagnostic)."""
    return bar_grid_report(load(ref, copy=False), expected)
