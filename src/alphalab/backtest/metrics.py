"""Mesures de performance et temoin aleatoire apparie.

Le temoin est la piece maitresse. Une esperance-R positive ne prouve rien seule :
il faut montrer qu'elle bat un tirage qui possede la MEME densite de trades, le MEME
biais achat/vente, la MEME geometrie de risque et la MEME distribution horaire. Sans
l'appariement horaire, une "strategie" qui ne fait qu'eviter les heures a spread eleve
bat le hasard sans contenir la moindre information sur la direction du prix.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from alphalab.backtest.engine import BacktestResult, ExecConfig, Order, SymbolData, run
from alphalab.types import FloatArray

#: Fournit (stop_distance, target_distance) pour un symbole et un indice de barre.
GeometryFn = Callable[[str, int], "tuple[float, float] | None"]

_BOOT_BLOCK = 500


def bootstrap_ci(
    r: FloatArray, *, n_boot: int = 10_000, seed: int = 12345, alpha: float = 0.05
) -> tuple[float, float]:
    """Intervalle de confiance percentile de l'esperance-R, par bootstrap."""
    n = r.size
    if n == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    done = 0
    while done < n_boot:
        block = min(_BOOT_BLOCK, n_boot - done)
        idx = rng.integers(0, n, size=(block, n))
        means[done : done + block] = r[idx].mean(axis=1)
        done += block
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi))


def equity_curve(trades: pd.DataFrame) -> pd.Series:
    """Courbe d'equite cumulee en R ponderes, indexee par date de sortie."""
    if trades.empty:
        return pd.Series(dtype=float)
    ordered = trades.sort_values("exit_ts", kind="stable")
    return pd.Series(
        ordered["r_weighted"].cumsum().to_numpy(dtype=float),
        index=pd.DatetimeIndex(ordered["exit_ts"]),
        name="equite_R",
    )


def max_drawdown(equity: pd.Series) -> float:
    """Plus forte perte depuis un sommet, en R. Valeur negative ou nulle."""
    if equity.empty:
        return 0.0
    values = equity.to_numpy(dtype=float)
    return float((values - np.maximum.accumulate(values)).min())


def matched_random_orders(
    reference: Sequence[Order],
    market: Mapping[str, SymbolData],
    rng: np.random.Generator,
    *,
    geometry: GeometryFn | None = None,
) -> list[Order]:
    """Ordres aleatoires apparies au jeu de reference.

    Pour chaque ordre reel on tire une barre au hasard du MEME symbole et de la MEME
    heure UTC, en conservant sens, echeance et geometrie de risque. Seul le choix du
    moment est randomise : c'est exactement ce que la strategie pretend savoir faire.
    """
    # Index des barres exploitables par (symbole, heure), calcule une fois.
    buckets: dict[tuple[str, int], FloatArray] = {}
    for symbol, data in market.items():
        hours = pd.DatetimeIndex(data.index).hour.to_numpy()
        usable = np.arange(len(data.index) - 1)  # il faut une barre suivante pour entrer
        for hour in np.unique(hours[:-1]):
            buckets[(symbol, int(hour))] = usable[hours[:-1] == hour]

    out: list[Order] = []
    for order in reference:
        found = market.get(order.symbol)
        if found is None:
            continue
        data = found
        key = (order.symbol, int(pd.Timestamp(order.signal_ts).hour))
        pool = buckets.get(key)
        if pool is None or pool.size == 0:
            continue
        j = int(pool[rng.integers(0, pool.size)])
        stop, target = order.stop_distance, order.target_distance
        if geometry is not None:
            geo = geometry(order.symbol, j)
            if geo is None:
                continue
            stop, target = geo
            if not (np.isfinite(stop) and stop > 0 and np.isfinite(target) and target > 0):
                continue
        out.append(
            Order(
                symbol=order.symbol,
                signal_ts=data.index[j],
                direction=order.direction,
                stop_distance=stop,
                target_distance=target,
                max_hold=order.max_hold,
                risk_frac=order.risk_frac,
                tag="temoin",
            )
        )
    return out


def beat_random(
    result: BacktestResult,
    reference: Sequence[Order],
    market: Mapping[str, SymbolData],
    *,
    n_control: int = 200,
    seed: int = 12345,
    cfg: ExecConfig | None = None,
    geometry: GeometryFn | None = None,
) -> dict[str, float]:
    """Fraction des temoins apparies que la strategie bat en esperance-R."""
    if result.n_trades == 0:
        return {"bat_hasard": float("nan"), "hasard_moyen": float("nan"), "n_temoins": 0}
    actual = float(result.trades["r"].mean())
    cfg = cfg or result.config
    means: list[float] = []
    for k in range(n_control):
        rng = np.random.default_rng(seed + 1000 + k)
        rand_orders = matched_random_orders(reference, market, rng, geometry=geometry)
        if not rand_orders:
            continue
        res = run(rand_orders, market, cfg)
        if res.n_trades:
            means.append(float(res.trades["r"].mean()))
    if not means:
        return {
            "bat_hasard": float("nan"),
            "hasard_moyen": float("nan"),
            "p_value": float("nan"),
            "n_temoins": 0,
        }
    arr = np.asarray(means, dtype=float)
    # p-value de permutation : probabilite qu'un tirage apparie fasse aussi bien.
    # La correction (1+k)/(1+n) interdit une p-value nulle, qui pretendrait une
    # certitude que n tirages ne peuvent pas fournir.
    p_value = (1 + int((arr >= actual).sum())) / (1 + arr.size)
    return {
        "bat_hasard": float((actual > arr).mean()),
        "hasard_moyen": float(arr.mean()),
        "hasard_ecart_type": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "p_value": float(p_value),
        "n_temoins": int(arr.size),
    }


def describe(
    result: BacktestResult, *, n_boot: int = 10_000, seed: int = 12345
) -> dict[str, Any]:
    """Statistiques descriptives d'une execution."""
    trades = result.trades
    if trades.empty:
        return {
            "trades": 0,
            "R_moyen": float("nan"),
            "IC95": (float("nan"), float("nan")),
            "exclut_0": False,
        }
    r = trades["r"].to_numpy(dtype=float)
    lo, hi = bootstrap_ci(r, n_boot=n_boot, seed=seed)
    eq = equity_curve(trades)
    wins = int((r > 0).sum())
    gross_win = float(r[r > 0].sum())
    gross_loss = float(-r[r < 0].sum())
    return {
        "trades": int(r.size),
        "part_achats": round(float((trades["direction"] == 1).mean()), 3),
        "R_moyen": round(float(r.mean()), 4),
        "R_median": round(float(np.median(r)), 4),
        "R_ecart_type": round(float(r.std(ddof=1)), 4) if r.size > 1 else 0.0,
        "IC95": (round(lo, 4), round(hi, 4)),
        "exclut_0": bool(lo > 0 or hi < 0),
        "taux_reussite": round(wins / r.size, 4),
        "facteur_profit": round(gross_win / gross_loss, 3) if gross_loss > 0 else float("inf"),
        "R_total": round(float(trades["r_weighted"].sum()), 3),
        "drawdown_max_R": round(max_drawdown(eq), 3),
        "duree_mediane_barres": float(trades["bars_held"].median()),
        "taux_ambigu": round(result.ambiguity_rate, 4),
        "issues": {str(k): int(v) for k, v in trades["outcome"].value_counts().items()},
    }
