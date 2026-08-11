"""Registre : ce qui est reellement disponible sur disque.

Le systeme ne code aucune liste d'instruments en dur dans sa logique. Il regarde ce
qui est present dans `data/snapshots/` et s'adapte. Deposer les snapshots EURUSD suffit
donc a activer le facteur dollar et les familles inter-actifs, sans toucher au code.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from alphalab.config import SNAPSHOT_DIR, UNIVERSE, SymbolSpec
from alphalab.data.snapshot import SnapshotError, SnapshotRef


@dataclass(frozen=True, slots=True)
class Availability:
    """Ce dont on dispose pour un couple (symbole, timeframe)."""

    symbol: str
    timeframe: str
    bid: SnapshotRef | None
    ask: SnapshotRef | None

    @property
    def has_bid(self) -> bool:
        return self.bid is not None

    @property
    def has_real_spread(self) -> bool:
        """Vrai si BID et ASK sont presents : le spread reel est calculable."""
        return self.bid is not None and self.ask is not None


def _version_key(ref: SnapshotRef) -> tuple[int, str]:
    """Trie les versions `v1`, `v2`, ... numeriquement quand c'est possible."""
    raw = ref.version.lstrip("vV")
    return (int(raw), ref.version) if raw.isdigit() else (-1, ref.version)


@lru_cache(maxsize=1)
def _scan() -> dict[tuple[str, str, str], SnapshotRef]:
    """Indexe les snapshots presents, en gardant la version la plus recente."""
    found: dict[tuple[str, str, str], SnapshotRef] = {}
    if not SNAPSHOT_DIR.is_dir():
        return found
    for manifest_path in sorted(SNAPSHOT_DIR.glob("*.manifest.json")):
        basename = manifest_path.name.removesuffix(".manifest.json")
        try:
            ref = SnapshotRef.from_basename(basename)
        except SnapshotError:
            continue  # fichier etranger au schema de nommage : ignore
        if not ref.csv_path.is_file():
            continue
        key = (ref.symbol, ref.timeframe, ref.side)
        current = found.get(key)
        if current is None or _version_key(ref) > _version_key(current):
            found[key] = ref
    return found


def refresh() -> None:
    """Force un nouveau scan du disque (apres un gel de donnees)."""
    _scan.cache_clear()


def find(symbol: str, timeframe: str, side: str) -> SnapshotRef | None:
    """Reference du snapshot disponible, ou None."""
    return _scan().get((symbol, timeframe, side))


def require(symbol: str, timeframe: str, side: str) -> SnapshotRef:
    """Comme `find`, mais leve une erreur explicite et actionnable si absent."""
    ref = find(symbol, timeframe, side)
    if ref is None:
        raise SnapshotError(
            f"Aucun snapshot {symbol} {timeframe} {side}. "
            f"Telechargez-le : alphalab freeze {symbol} --tf {timeframe} --side {side}"
        )
    return ref


def availability(symbol: str, timeframe: str) -> Availability:
    return Availability(
        symbol=symbol,
        timeframe=timeframe,
        bid=find(symbol, timeframe, "BID"),
        ask=find(symbol, timeframe, "ASK"),
    )


def available_symbols() -> list[str]:
    """Symboles ayant au moins un snapshot, tries."""
    return sorted({symbol for symbol, _, _ in _scan()})


def available_timeframes(symbol: str) -> list[str]:
    return sorted({tf for sym, tf, _ in _scan() if sym == symbol})


def tradable(timeframe: str) -> list[str]:
    """Symboles utilisables en backtest a ce timeframe : BID et ASK presents.

    Sans les deux cotes, le spread reel n'est pas mesurable et le resultat serait
    fonde sur un cout suppose. Le projet a deja montre qu'un cout suppose peut
    transformer une strategie perdante en gagnante apparente.
    """
    return sorted(
        {
            symbol
            for symbol, tf, _ in _scan()
            if tf == timeframe and availability(symbol, tf).has_real_spread
        }
    )


def spec(symbol: str) -> SymbolSpec:
    """Fiche d'un instrument. Les symboles inconnus recoivent une fiche neutre."""
    known = UNIVERSE.get(symbol)
    if known is not None:
        return known
    return SymbolSpec(symbol=symbol, label=symbol, dukascopy_id="", kind="unknown", usd_leg=0)


def summary() -> list[dict[str, object]]:
    """Etat du disque, pour affichage CLI et en-tete de rapport."""
    rows: list[dict[str, object]] = []
    for symbol in available_symbols():
        for tf in available_timeframes(symbol):
            avail = availability(symbol, tf)
            rows.append(
                {
                    "symbole": symbol,
                    "timeframe": tf,
                    "BID": avail.has_bid,
                    "ASK": avail.ask is not None,
                    "spread_reel": avail.has_real_spread,
                }
            )
    return rows
