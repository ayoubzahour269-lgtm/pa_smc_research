"""Construction du jeu d'apprentissage : candidats, features, labels.

Le label suit le principe de la **triple barriere** : chaque candidat se resout par
l'objectif, le stop ou l'echeance. Nul besoin de le reimplementer — le moteur de
backtest applique deja exactement ces trois barrieres, avec le cout reel. On lui
demande donc le resultat, et le label devient `R > 0`.

Deux consequences importantes de ce choix :

  - le label inclut le COUT. Un trade qui touche l'objectif mais dont le spread mange
    le gain compte comme perdant, ce qui est la verite economique ;
  - chaque ligne porte sa date d'entree ET sa date de sortie. C'est indispensable au
    decoupage purge : deux trades dont les fenetres se chevauchent ne sont pas des
    observations independantes.

La modelisation est une **meta-labellisation** : le modele ne predit pas la direction
(la famille la donne), il predit la probabilite que ce candidat-la gagne. C'est
exactement la quantite demandee par un plan de trading — "quelle est la probabilite que
cette position soit gagnante ?" — et non un score de confluence arbitraire.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from alphalab.alpha.base import AlphaFamily, Context
from alphalab.alpha.registry import family_label
from alphalab.backtest.engine import ExecConfig, SymbolData, run
from alphalab.types import IntArray

#: Colonnes de service, jamais fournies au modele comme variables explicatives.
META_COLUMNS = (
    "famille",
    "symbole",
    "signal_ts",
    "entry_ts",
    "exit_ts",
    "direction",
    "r",
    "label",
    "outcome",
)


def build(
    ctx: Context,
    families: Sequence[AlphaFamily],
    market: dict[str, SymbolData],
    *,
    cfg: ExecConfig | None = None,
    feature_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Une ligne par candidat execute, avec ses features au moment du signal.

    Les candidats sont executes famille par famille, SANS contrainte de concurrence
    croisee : on cherche a caracteriser chaque candidat pris isolement, pas a simuler
    un portefeuille. Le portefeuille est l'affaire du moteur de risque, en aval.
    """
    cfg = cfg or ExecConfig(max_concurrent=1, max_per_symbol=1)
    columns = list(feature_columns) if feature_columns else list(ctx.features.columns)

    frames: list[pd.DataFrame] = []
    for family in families:
        orders = family.generate(ctx)
        if not orders:
            continue
        result = run(orders, market, cfg)
        if result.trades.empty:
            continue

        trades = result.trades
        signal_ts = pd.DatetimeIndex(trades["signal_ts"])
        features = ctx.features.reindex(signal_ts)[columns].reset_index(drop=True)

        rows = pd.DataFrame(
            {
                "famille": family_label(family),
                "symbole": ctx.symbol,
                "signal_ts": trades["signal_ts"].to_numpy(),
                "entry_ts": trades["entry_ts"].to_numpy(),
                "exit_ts": trades["exit_ts"].to_numpy(),
                "direction": trades["direction"].to_numpy(),
                "r": trades["r"].to_numpy(),
                "label": (trades["r"].to_numpy() > 0).astype(int),
                "outcome": trades["outcome"].to_numpy(),
            }
        )
        frames.append(pd.concat([rows, features], axis=1))

    if not frames:
        return pd.DataFrame(columns=[*META_COLUMNS, *columns])

    out = pd.concat(frames, ignore_index=True)
    return out.sort_values("entry_ts", kind="stable").reset_index(drop=True)


def feature_matrix(
    data: pd.DataFrame, *, extra_categorical: Sequence[str] = ("famille",)
) -> tuple[pd.DataFrame, IntArray]:
    """Sepаre variables explicatives et label, en encodant la famille d'origine.

    La famille est encodee car elle porte de l'information : savoir qu'un candidat vient
    d'un balayage de liquidite plutot que d'un gap change ses chances, independamment
    des autres features.
    """
    y: IntArray = data["label"].to_numpy(dtype=np.int64)
    numeric = data.drop(columns=[c for c in META_COLUMNS if c in data.columns])
    numeric = numeric.select_dtypes(include=[np.number, bool]).astype(float)

    parts = [numeric]
    for column in extra_categorical:
        if column in data.columns:
            parts.append(pd.get_dummies(data[column], prefix=column, dtype=float))
    X = pd.concat(parts, axis=1)

    # Les NaN de periode de chauffe sont remplaces par la mediane de la COLONNE, ce qui
    # est acceptable ici : ces colonnes sont deja toutes causales, et l'imputation ne
    # reintroduit pas d'information future puisqu'elle est refaite sur chaque pli
    # d'apprentissage (voir walkforward.fit_predict).
    return X, y
