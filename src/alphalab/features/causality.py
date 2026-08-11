"""Verification mecanique de l'absence d'anticipation.

Le contrat de tout le paquet `features` tient en une ligne :

    f(df)[t] == f(df.loc[:t])[t]     pour tout t

Ce module le VERIFIE au lieu de le promettre. Le principe : recalculer la feature sur
un historique tronque a `t`, puis exiger que la valeur en `t` soit identique a celle
obtenue sur l'historique complet. Si elle differe, la feature a utilise de l'information
posterieure a `t` — elle est inexploitable en production, et tout backtest qui s'appuie
dessus est faux.

C'est le test le plus important du depot. Les quatre pieges classiques du domaine :

  - un pivot fractal est expose a sa date de formation au lieu de sa date de
    confirmation (`W` barres plus tard) ;
  - un range de seance est expose pendant la seance au lieu d'apres sa cloture ;
  - une zone est marquee "validee" retroactivement une fois qu'on sait qu'elle a tenu ;
  - une serie macro revisee est utilisee a sa valeur finale plutot qu'a son millesime.

Utilisation lors de l'ajout d'une feature :

    from alphalab.features.causality import assert_causal
    assert_causal(ma_feature, df)
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TypeAlias

import numpy as np
import pandas as pd

#: Une feature transforme une trame OHLCV en colonnes alignees sur le meme index.
FeatureFn: TypeAlias = Callable[[pd.DataFrame], "pd.Series | pd.DataFrame"]

DEFAULT_PROBES = 25
DEFAULT_MIN_HISTORY = 400


def _as_frame(obj: pd.Series | pd.DataFrame) -> pd.DataFrame:
    return obj.to_frame() if isinstance(obj, pd.Series) else obj


def _values_match(a: object, b: object, *, rtol: float, atol: float) -> bool:
    """Egalite tolerante aux NaN et aux flottants."""
    a_null = a is None or (isinstance(a, float) and np.isnan(a)) or a is pd.NaT
    b_null = b is None or (isinstance(b, float) and np.isnan(b)) or b is pd.NaT
    if a_null or b_null:
        return bool(a_null and b_null)
    if isinstance(a, bool | np.bool_) or isinstance(b, bool | np.bool_):
        return bool(a) == bool(b)
    numeric = int | float | np.integer | np.floating
    if isinstance(a, numeric) and isinstance(b, numeric):
        return bool(np.isclose(float(a), float(b), rtol=rtol, atol=atol))
    return bool(a == b)


def probe_timestamps(
    df: pd.DataFrame,
    *,
    n_probes: int = DEFAULT_PROBES,
    min_history: int = DEFAULT_MIN_HISTORY,
    seed: int = 0,
) -> list[pd.Timestamp]:
    """Horodatages de controle, tires au hasard au-dela de la periode de chauffe.

    On evite le debut de l'historique : les indicateurs y valent NaN des deux cotes,
    ce qui ferait passer le test sans rien prouver.
    """
    n = len(df)
    if n <= min_history + 2:
        raise ValueError(
            f"Historique trop court pour un controle utile : {n} barres, "
            f"il en faut plus de {min_history + 2}"
        )
    rng = np.random.default_rng(seed)
    positions = rng.choice(
        np.arange(min_history, n), size=min(n_probes, n - min_history), replace=False
    )
    return [df.index[int(p)] for p in np.sort(positions)]


def check_causal(
    fn: FeatureFn,
    df: pd.DataFrame,
    *,
    n_probes: int = DEFAULT_PROBES,
    min_history: int = DEFAULT_MIN_HISTORY,
    seed: int = 0,
    rtol: float = 1e-9,
    atol: float = 1e-12,
    columns: Sequence[str] | None = None,
) -> list[str]:
    """Renvoie la liste des violations de causalite. Liste vide = feature conforme.

    Chaque violation nomme la colonne, l'horodatage et les deux valeurs en desaccord,
    de sorte que le diagnostic ne demande aucune investigation supplementaire.
    """
    full = _as_frame(fn(df))
    checked = list(columns) if columns is not None else list(full.columns)
    violations: list[str] = []

    for ts in probe_timestamps(df, n_probes=n_probes, min_history=min_history, seed=seed):
        truncated = _as_frame(fn(df.loc[:ts]))
        if ts not in truncated.index:
            violations.append(f"{ts}: l'horodatage disparait du resultat tronque")
            continue
        row_full = full.loc[ts]
        row_trunc = truncated.loc[ts]
        for col in checked:
            if col not in truncated.columns:
                violations.append(f"{col} @ {ts}: colonne absente du resultat tronque")
                continue
            a, b = row_full[col], row_trunc[col]
            if not _values_match(a, b, rtol=rtol, atol=atol):
                violations.append(
                    f"{col} @ {ts}: historique complet = {a!r}, tronque = {b!r} "
                    "-> la feature utilise de l'information future"
                )
    return violations


def assert_causal(
    fn: FeatureFn,
    df: pd.DataFrame,
    *,
    label: str = "",
    n_probes: int = DEFAULT_PROBES,
    min_history: int = DEFAULT_MIN_HISTORY,
    seed: int = 0,
    rtol: float = 1e-9,
    atol: float = 1e-12,
    columns: Sequence[str] | None = None,
    max_reported: int = 8,
) -> None:
    """Leve `AssertionError` si la feature anticipe."""
    violations = check_causal(
        fn,
        df,
        n_probes=n_probes,
        min_history=min_history,
        seed=seed,
        rtol=rtol,
        atol=atol,
        columns=columns,
    )
    if violations:
        head = "\n  ".join(violations[:max_reported])
        hidden = len(violations) - max_reported
        extra = f"\n  ... et {hidden} autres" if hidden > 0 else ""
        raise AssertionError(
            f"ANTICIPATION DETECTEE{f' dans {label}' if label else ''} — "
            f"{len(violations)} violation(s) :\n  {head}{extra}"
        )


def synthetic_ohlcv(
    n: int = 3000,
    *,
    seed: int = 0,
    start: str = "2020-01-06 00:00",
    freq: str = "15min",
    start_price: float = 2000.0,
    vol: float = 0.0015,
) -> pd.DataFrame:
    """Marche synthetique reproductible pour les controles de causalite.

    Une marche aleatoire suffit : la causalite est une propriete du CODE, pas des
    donnees. Utiliser des donnees synthetiques permet de faire tourner ces tests en
    CI, sans acces au moindre fournisseur de marche.
    """
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0, vol, n)
    close = start_price * np.exp(np.cumsum(steps))
    open_ = np.concatenate([[start_price], close[:-1]])
    spread = np.abs(rng.normal(0.0, vol, n)) * close
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    index = pd.date_range(start, periods=n, freq=freq, tz="UTC", name="timestamp")
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.abs(rng.normal(1.0, 0.3, n)),
        },
        index=index,
    )
