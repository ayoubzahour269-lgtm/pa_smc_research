"""Alias de types partages."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

#: Tableau de flottants 64 bits — la representation de travail de tout le moteur.
FloatArray = npt.NDArray[np.float64]

#: Tableau de booleens (masques de features).
BoolArray = npt.NDArray[np.bool_]

#: Tableau d'entiers (indices de barres, sens de position).
IntArray = npt.NDArray[np.int64]
