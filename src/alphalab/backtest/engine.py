"""Moteur de backtest : portefeuille, multi-symboles, cout reel de premiere classe.

Semantique d'execution, volontairement conservatrice :

  - un signal emis sur la barre `i` s'execute a l'OUVERTURE de la barre `i+1`
    (aucune execution au prix qui a servi a decider) ;
  - le cout est celui de la barre d'entree, preleve integralement a l'entree ;
  - dans une barre ou stop et objectif sont tous deux atteignables, on retient le
    STOP. On ne peut pas savoir lequel a ete touche en premier sans donnees tick ;
    choisir l'issue defavorable est la seule hypothese qui ne fabrique pas d'edge.
    Le taux de ces barres ambigues est compte et publie : c'est la mesure honnete de
    l'incertitude du backtest, et elle sera decisive pour juger tout resultat M1/M5 ;
  - a l'echeance, sortie a la cloture de la derniere barre detenue.

Le moteur ne sait rien des indicateurs, des familles d'alpha ni des grades. Il recoit
des ordres deja formes et les execute. C'est ce qui permet de comparer sur une meme
echelle des approches heterogenes.
"""

from __future__ import annotations

import heapq
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final, Literal

import numpy as np
import pandas as pd

from alphalab.data.schema import validate_ohlcv
from alphalab.types import FloatArray

IntrabarRule = Literal["stop_first", "target_first"]

#: Issues possibles d'un trade.
EXIT_TARGET: Final = "objectif"
EXIT_STOP: Final = "stop"
EXIT_TIME: Final = "echeance"
EXIT_SCHEDULED: Final = "horaire"

TRADE_COLUMNS: Final[tuple[str, ...]] = (
    "symbol",
    "tag",
    "direction",
    "signal_ts",
    "entry_ts",
    "entry",
    "exit_ts",
    "exit",
    "bars_held",
    "risk_unit",
    "cost",
    "r",
    "risk_frac",
    "r_weighted",
    "outcome",
    "ambiguous",
)


@dataclass(frozen=True, slots=True)
class ExitPolicy:
    """Regles de sortie autres que le stop initial et l'objectif fixe.

    Cette dimension a longtemps manque au projet : 48 configurations ont ete evaluees
    avec une seule et meme regle de sortie (stop fixe, objectif fixe, echeance). Or la
    litterature de suivi de tendance est constante sur ce point — la sortie pese autant
    que l'entree, et une monoculture de sortie peut faire paraitre mauvaise n'importe
    quelle entree.

    Toutes les distances sont exprimees en multiples de 1R (la distance du stop
    initial), ce qui les rend comparables entre instruments et entre volatilites.
    """

    #: Distance du stop suiveur derriere l'extreme favorable, en R. None = pas de suivi.
    trailing_r: float | None = None
    #: Gain (en R) a partir duquel le stop remonte au prix d'entree. None = jamais.
    breakeven_at_r: float | None = None
    #: Heure UTC a laquelle la position est fermee d'office. None = jamais.
    close_at_hour: int | None = None

    def __post_init__(self) -> None:
        if self.trailing_r is not None and self.trailing_r <= 0:
            raise ValueError(f"trailing_r doit etre > 0, recu {self.trailing_r}")
        if self.breakeven_at_r is not None and self.breakeven_at_r <= 0:
            raise ValueError(f"breakeven_at_r doit etre > 0, recu {self.breakeven_at_r}")
        if self.close_at_hour is not None and not 0 <= self.close_at_hour <= 23:
            raise ValueError(f"close_at_hour doit etre entre 0 et 23, recu {self.close_at_hour}")

    @property
    def label(self) -> str:
        parts = []
        if self.trailing_r is not None:
            parts.append(f"suiveur{self.trailing_r}R")
        if self.breakeven_at_r is not None:
            parts.append(f"seuil{self.breakeven_at_r}R")
        if self.close_at_hour is not None:
            parts.append(f"cloture{self.close_at_hour}h")
        return "+".join(parts) if parts else "fixe"


#: Sortie de reference : stop fixe, objectif fixe, echeance. Celle des campagnes S6-S9.
FIXED_EXIT: Final = ExitPolicy()


@dataclass(frozen=True, slots=True)
class Order:
    """Intention d'entree produite par une famille d'alpha.

    Les distances sont en unites de prix, pas en multiples d'ATR : c'est a la famille
    de decider comment elle dimensionne son risque, et au moteur de l'executer tel quel.
    """

    symbol: str
    signal_ts: pd.Timestamp
    direction: int  # +1 achat, -1 vente
    stop_distance: float  # 1R, en unites de prix, strictement positif
    target_distance: float  # distance de l'objectif, en unites de prix
    max_hold: int  # nombre maximal de barres detenues
    risk_frac: float = 1.0  # fraction de risque unitaire (paliers A/B/C)
    tag: str = ""
    exit_policy: ExitPolicy = FIXED_EXIT

    def __post_init__(self) -> None:
        if self.direction not in (1, -1):
            raise ValueError(f"direction doit valoir +1 ou -1, recu {self.direction}")
        if not np.isfinite(self.stop_distance) or self.stop_distance <= 0:
            raise ValueError(f"stop_distance doit etre fini et > 0, recu {self.stop_distance}")
        if not np.isfinite(self.target_distance) or self.target_distance <= 0:
            raise ValueError(f"target_distance doit etre fini et > 0, recu {self.target_distance}")
        if self.max_hold < 1:
            raise ValueError(f"max_hold doit valoir au moins 1, recu {self.max_hold}")


@dataclass(frozen=True, slots=True)
class ExecConfig:
    """Contraintes d'execution du portefeuille."""

    max_concurrent: int = 3
    max_per_symbol: int = 1
    intrabar: IntrabarRule = "stop_first"
    #: Cout de repli quand le spread reel manque sur la barre d'entree, exprime en
    #: fraction de 1R. Defaut volontairement penalisant.
    fallback_cost_frac_of_risk: float = 0.02

    def __post_init__(self) -> None:
        if self.max_concurrent < 1:
            raise ValueError("max_concurrent doit valoir au moins 1")
        if self.max_per_symbol < 1:
            raise ValueError("max_per_symbol doit valoir au moins 1")


@dataclass(slots=True)
class SymbolData:
    """Prix et couts d'un symbole, prets pour l'execution."""

    symbol: str
    index: pd.DatetimeIndex
    open: FloatArray
    high: FloatArray
    low: FloatArray
    close: FloatArray
    cost: FloatArray  # cout par barre, en unites de prix ; NaN autorise
    _pos: dict[pd.Timestamp, int] = field(default_factory=dict, repr=False)

    @classmethod
    def from_frame(
        cls, symbol: str, df: pd.DataFrame, cost: FloatArray | pd.Series | None = None
    ) -> SymbolData:
        validate_ohlcv(df, name=symbol, require_volume=False)
        n = len(df)
        cost_arr: FloatArray
        if cost is None:
            cost_arr = np.full(n, np.nan, dtype=np.float64)
        elif isinstance(cost, pd.Series):
            cost_arr = cost.reindex(df.index).to_numpy(dtype=np.float64)
        else:
            cost_arr = np.asarray(cost, dtype=np.float64)
        if cost_arr.shape != (n,):
            raise ValueError(
                f"{symbol}: le cout doit avoir {n} valeurs (une par barre), recu {cost_arr.shape}"
            )
        index = pd.DatetimeIndex(df.index)
        return cls(
            symbol=symbol,
            index=index,
            open=df["open"].to_numpy(dtype=float),
            high=df["high"].to_numpy(dtype=float),
            low=df["low"].to_numpy(dtype=float),
            close=df["close"].to_numpy(dtype=float),
            cost=cost_arr,
            _pos={ts: i for i, ts in enumerate(index)},
        )

    def position_of(self, ts: pd.Timestamp) -> int | None:
        return self._pos.get(ts)


@dataclass(frozen=True, slots=True)
class BacktestResult:
    """Resultat d'une execution : le journal des trades et ce qui n'a pas eu lieu."""

    trades: pd.DataFrame
    n_orders: int
    skipped: dict[str, int]
    config: ExecConfig

    @property
    def n_trades(self) -> int:
        return int(len(self.trades))

    @property
    def r(self) -> FloatArray:
        """Esperances-R brutes, une par trade (hors ponderation de risque)."""
        if self.trades.empty:
            return np.empty(0, dtype=np.float64)
        return self.trades["r"].to_numpy(dtype=np.float64)

    @property
    def ambiguity_rate(self) -> float:
        """Part des trades dont la barre de sortie touchait stop ET objectif.

        Un taux eleve signifie que le resultat depend fortement de l'hypothese
        intrabarre, donc qu'il est peu fiable sans donnees tick.
        """
        if self.trades.empty:
            return 0.0
        return float(self.trades["ambiguous"].mean())

    def summary(self) -> dict[str, object]:
        out: dict[str, object] = {
            "ordres": self.n_orders,
            "trades": self.n_trades,
            "ignores": dict(self.skipped),
            "taux_ambigu": round(self.ambiguity_rate, 4),
        }
        if not self.trades.empty:
            counts = self.trades["outcome"].value_counts().to_dict()
            out["issues"] = {str(k): int(v) for k, v in counts.items()}
            out["R_moyen"] = round(float(self.trades["r"].mean()), 4)
        return out


def _simulate_one(
    order: Order, data: SymbolData, cfg: ExecConfig
) -> tuple[dict[str, object], str] | tuple[None, str]:
    """Execute un ordre isolement. Renvoie (ligne de journal, "") ou (None, raison)."""
    i = data.position_of(order.signal_ts)
    if i is None:
        return None, "horodatage absent de la grille"
    entry_idx = i + 1
    n = len(data.index)
    if entry_idx >= n:
        return None, "pas de barre suivante pour entrer"

    entry = float(data.open[entry_idx])
    if not np.isfinite(entry):
        return None, "prix d'entree non fini"

    d = order.direction
    risk_unit = float(order.stop_distance)
    stop = entry - d * risk_unit
    target = entry + d * order.target_distance

    raw_cost = float(data.cost[entry_idx])
    cost = raw_cost if np.isfinite(raw_cost) else cfg.fallback_cost_frac_of_risk * risk_unit

    last_idx = min(entry_idx + order.max_hold - 1, n - 1)
    exit_idx = last_idx
    exit_price = float(data.close[last_idx])
    outcome = EXIT_TIME
    ambiguous = False

    policy = order.exit_policy
    best = entry  # extreme favorable atteint, mis a jour en FIN de barre

    for j in range(entry_idx, last_idx + 1):
        hi, lo = data.high[j], data.low[j]

        # Sortie a heure fixe : evaluee avant les barrieres, car elle represente une
        # decision prise a l'ouverture de la barre (ex. "je ne garde pas de position
        # apres la cloture cash").
        scheduled = (
            policy.close_at_hour is not None
            and j > entry_idx
            and int(data.index[j].hour) == policy.close_at_hour
        )
        if scheduled:
            exit_idx, exit_price, outcome = j, float(data.close[j]), EXIT_SCHEDULED
            break

        hit_stop = lo <= stop if d == 1 else hi >= stop
        hit_target = hi >= target if d == 1 else lo <= target
        if hit_stop and hit_target:
            ambiguous = True
            if cfg.intrabar == "stop_first":
                exit_idx, exit_price, outcome = j, stop, EXIT_STOP
            else:
                exit_idx, exit_price, outcome = j, target, EXIT_TARGET
            break
        if hit_stop:
            exit_idx, exit_price, outcome = j, stop, EXIT_STOP
            break
        if hit_target:
            exit_idx, exit_price, outcome = j, target, EXIT_TARGET
            break

        # Mise a jour des niveaux APRES le test des barrieres, donc pour la barre
        # SUIVANTE. Ce decalage est la seule facon correcte de simuler un stop suiveur
        # sur des barres : le mettre a jour avant le test permettrait a une barre d'etre
        # stoppee par un niveau derive de son propre extreme — un stop qui se declenche
        # grace a un prix qu'il n'avait pas encore vu.
        if policy.trailing_r is not None or policy.breakeven_at_r is not None:
            best = max(best, hi) if d == 1 else min(best, lo)
            if policy.breakeven_at_r is not None:
                progress = d * (best - entry) / risk_unit
                if progress >= policy.breakeven_at_r:
                    stop = max(stop, entry) if d == 1 else min(stop, entry)
            if policy.trailing_r is not None:
                trail = best - d * policy.trailing_r * risk_unit
                stop = max(stop, trail) if d == 1 else min(stop, trail)

    r = (d * (exit_price - entry) - cost) / risk_unit
    row: dict[str, object] = {
        "symbol": order.symbol,
        "tag": order.tag,
        "direction": d,
        "signal_ts": order.signal_ts,
        "entry_ts": data.index[entry_idx],
        "entry": entry,
        "exit_ts": data.index[exit_idx],
        "exit": exit_price,
        "bars_held": int(exit_idx - entry_idx + 1),
        "risk_unit": risk_unit,
        "cost": cost,
        "r": float(r),
        "risk_frac": float(order.risk_frac),
        "r_weighted": float(r * order.risk_frac),
        "outcome": outcome,
        "ambiguous": ambiguous,
    }
    return row, ""


def run(
    orders: Iterable[Order],
    market: Mapping[str, SymbolData],
    cfg: ExecConfig | None = None,
) -> BacktestResult:
    """Execute une sequence d'ordres sous contrainte de portefeuille.

    Les ordres sont traites dans l'ordre chronologique des signaux. Un ordre qui
    violerait une contrainte (trop de positions ouvertes, symbole deja engage) est
    ignore et compte : le journal des refus fait partie du resultat, car il decrit
    ce que la contrainte a coute en opportunites.
    """
    cfg = cfg or ExecConfig()
    ordered: Sequence[Order] = sorted(orders, key=lambda o: (o.signal_ts, o.symbol, o.tag))

    rows: list[dict[str, object]] = []
    skipped: dict[str, int] = {}

    # Positions ouvertes, dans un tas trie par date de sortie. Les ordres etant traites
    # par date de signal croissante, leurs dates d'entree le sont aussi : une position
    # deja retenue chevauche donc l'entree visee si et seulement si elle n'est pas
    # encore sortie. Il suffit d'expulser les positions closes en tete de tas, ce qui
    # remplace un balayage de toutes les positions par un cout logarithmique. Sur une
    # campagne d'exploration (des milliers de rejeux), la difference est decisive.
    open_positions: list[tuple[pd.Timestamp, str]] = []
    per_symbol: Counter[str] = Counter()

    def note(reason: str) -> None:
        skipped[reason] = skipped.get(reason, 0) + 1

    n_orders = 0
    for order in ordered:
        n_orders += 1
        data = market.get(order.symbol)
        if data is None:
            note(f"symbole absent du marche fourni ({order.symbol})")
            continue

        i = data.position_of(order.signal_ts)
        if i is None or i + 1 >= len(data.index):
            note("horodatage inexploitable")
            continue
        entry_ts = data.index[i + 1]

        while open_positions and open_positions[0][0] < entry_ts:
            _, closed_symbol = heapq.heappop(open_positions)
            per_symbol[closed_symbol] -= 1

        if len(open_positions) >= cfg.max_concurrent:
            note("plafond de positions simultanees")
            continue
        if per_symbol[order.symbol] >= cfg.max_per_symbol:
            note("plafond de positions par symbole")
            continue

        row, reason = _simulate_one(order, data, cfg)
        if row is None:
            note(reason)
            continue
        rows.append(row)
        exit_ts: pd.Timestamp = row["exit_ts"]  # type: ignore[assignment]
        heapq.heappush(open_positions, (exit_ts, order.symbol))
        per_symbol[order.symbol] += 1

    trades = pd.DataFrame(rows, columns=list(TRADE_COLUMNS))
    if not trades.empty:
        trades = trades.sort_values("entry_ts", kind="stable").reset_index(drop=True)
    return BacktestResult(trades=trades, n_orders=n_orders, skipped=skipped, config=cfg)
