"""Validation croisee glissante, purgee et embargoed.

Un decoupage naif fuit, pour une raison specifique aux donnees financieres : les trades
se CHEVAUCHENT dans le temps. Un trade d'apprentissage ouvert le 3 et clos le 5 partage
son evolution de prix avec un trade de test ouvert le 4. Le modele a donc deja vu, sous
une autre etiquette, une partie de ce qu'on lui demande de predire — et le score de test
devient optimiste sans que rien ne le signale.

Deux protections, toutes deux necessaires :

  - **purge** : on retire de l'apprentissage tout trade dont la fenetre [entree, sortie]
    chevauche celle d'un trade de test ;
  - **embargo** : on retire en plus une bande de temps juste apres le pli de test, car
    l'autocorrelation des rendements fait qu'une observation immediatement posterieure
    reste informative sur la periode de test.

Le decoupage est **glissant vers l'avant** : on n'apprend jamais sur des donnees
posterieures a ce qu'on predit. Un k-fold melange le passe et le futur, ce qui n'a aucun
sens pour une serie temporelle.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd

from alphalab.types import IntArray

#: Bande d'embargo par defaut. Definie au niveau module pour ne pas evaluer un appel
#: de fonction a chaque definition de signature.
DEFAULT_EMBARGO = pd.Timedelta(days=5)


@dataclass(frozen=True, slots=True)
class Split:
    """Un pli : indices d'apprentissage et indices de test."""

    train: IntArray
    test: IntArray
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    @property
    def n_train(self) -> int:
        return int(self.train.size)

    @property
    def n_test(self) -> int:
        return int(self.test.size)


def purged_walk_forward(
    entry_ts: pd.Series,
    exit_ts: pd.Series,
    *,
    n_splits: int = 5,
    embargo: pd.Timedelta = DEFAULT_EMBARGO,
    min_train: int = 200,
) -> Iterator[Split]:
    """Genere des plis glissants, purges et embargoed.

    Les observations sont supposees triees par date d'entree. Chaque pli de test est une
    tranche temporelle consecutive ; l'apprentissage n'utilise que des observations
    ANTERIEURES, dont la fenetre ne chevauche ni le test ni sa bande d'embargo.
    """
    entries = pd.DatetimeIndex(entry_ts)
    exits = pd.DatetimeIndex(exit_ts)
    n = len(entries)
    if n == 0:
        return

    # Controle de monotonie, et non d'egalite a un argsort : plusieurs familles peuvent
    # se declencher sur la MEME barre, et un tri instable renverrait alors une
    # permutation differente de l'identite pour des donnees pourtant bien triees.
    if not entries.is_monotonic_increasing:
        raise ValueError("Les observations doivent etre triees par date d'entree")

    boundaries = np.linspace(0, n, n_splits + 1, dtype=int)
    for k in range(1, n_splits + 1):
        test_lo, test_hi = boundaries[k - 1], boundaries[k]
        if test_hi - test_lo == 0:
            continue
        test_idx = np.arange(test_lo, test_hi)
        test_start = entries[test_lo]
        test_end = exits[test_idx].max()

        # Apprentissage : uniquement le passe, et uniquement les trades DEJA CLOS avant
        # le debut du test. C'est la purge.
        candidate = np.arange(0, test_lo)
        if candidate.size == 0:
            continue
        closed_before_test = exits[candidate] < test_start
        train_idx = candidate[closed_before_test]

        # Embargo : on ecarte aussi ce qui touche la bande precedant immediatement le
        # test, ou l'autocorrelation rend l'information encore partagee.
        if embargo > pd.Timedelta(0) and train_idx.size:
            keep = exits[train_idx] < (test_start - embargo)
            train_idx = train_idx[keep]

        if train_idx.size < min_train:
            continue
        yield Split(
            train=train_idx.astype(np.int64),
            test=test_idx.astype(np.int64),
            test_start=test_start,
            test_end=test_end,
        )


def coverage(splits: list[Split], n_total: int) -> dict[str, float]:
    """Part des observations effectivement evaluees hors echantillon.

    Utile pour verifier qu'un decoupage trop severe ne laisse pas 90 % des donnees sans
    prediction — auquel cas la calibration porterait sur trop peu de choses pour valoir.
    """
    if not splits or n_total == 0:
        return {"plis": 0, "couverture_test": 0.0, "apprentissage_median": 0.0}
    tested = np.unique(np.concatenate([s.test for s in splits]))
    return {
        "plis": len(splits),
        "couverture_test": round(float(tested.size / n_total), 4),
        "apprentissage_median": float(np.median([s.n_train for s in splits])),
    }
