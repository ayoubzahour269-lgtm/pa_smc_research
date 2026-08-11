"""Indicateurs techniques — source unique de verite.

CONTRAT CAUSAL, valable pour toute fonction de ce module et de `features/` :

    f(df)[t] == f(df.loc[:t])[t]     pour tout t

Autrement dit, la valeur en `t` ne depend que des barres <= t. C'est verifie
mecaniquement par `tests/antilookahead/`, pas seulement promis ici.

Ce module existe aussi pour une raison prosaique : dans la version precedente du
projet, l'ATR etait reimplemente en quatre endroits differents. Deux d'entre eux
utilisaient une moyenne simple, un troisieme un lissage de Wilder — les resultats
n'etaient donc pas comparables entre eux sans que personne ne le sache.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd

DEFAULT_ATR_LEN: Final[int] = 14


def true_range(df: pd.DataFrame) -> pd.Series:
    """True range de Wilder : max(h-l, |h-c_prec|, |l-c_prec|)."""
    prev_close = df["close"].shift(1)
    ranges = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1).rename("true_range")


def atr(df: pd.DataFrame, length: int = DEFAULT_ATR_LEN, *, wilder: bool = True) -> pd.Series:
    """Average true range.

    `wilder=True` (defaut) applique le lissage exponentiel de Wilder (alpha = 1/n),
    qui est la definition d'origine. `wilder=False` donne la moyenne mobile simple.
    Les deux sont proposees explicitement pour qu'aucun appelant n'ait a redevenir
    ce choix par accident.
    """
    tr = true_range(df)
    if wilder:
        out = tr.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()
    else:
        out = tr.rolling(length).mean()
    return out.rename(f"atr{length}")


def ema(series: pd.Series, span: int) -> pd.Series:
    """Moyenne mobile exponentielle (sans reponderation retroactive)."""
    return series.ewm(span=span, adjust=False, min_periods=span).mean().rename(f"ema{span}")


def sma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(length).mean().rename(f"sma{length}")


def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    """RSI de Wilder, borne 0-100."""
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    # Perte moyenne nulle = hausse ininterrompue : RSI vaut 100 par definition.
    out = out.where(~((avg_loss == 0) & (avg_gain > 0)), 100.0)
    result: pd.Series = out.rename(f"rsi{length}")
    return result


def adx(df: pd.DataFrame, length: int = 14) -> pd.DataFrame:
    """ADX et ses composantes directionnelles (+DI, -DI).

    Sert de mesure de "force de tendance" pour conditionner les familles de
    momentum : une cassure nue a deja ete rejetee sur ce projet, la question ouverte
    est de savoir si elle se comporte differemment selon le regime.
    """
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)

    alpha = 1.0 / length
    tr_s = true_range(df).ewm(alpha=alpha, adjust=False, min_periods=length).mean()
    plus_di = 100.0 * plus_dm.ewm(alpha=alpha, adjust=False, min_periods=length).mean() / tr_s
    minus_di = 100.0 * minus_dm.ewm(alpha=alpha, adjust=False, min_periods=length).mean() / tr_s

    denom = (plus_di + minus_di).replace(0.0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / denom
    return pd.DataFrame(
        {
            "plus_di": plus_di,
            "minus_di": minus_di,
            "adx": dx.ewm(alpha=alpha, adjust=False, min_periods=length).mean(),
        }
    )


def rolling_high(df: pd.DataFrame, length: int) -> pd.Series:
    """Plus haut des `length` barres PRECEDENTES (barre courante exclue).

    L'exclusion est le point important : inclure la barre courante permettrait a une
    "cassure" de se comparer a elle-meme, ce qui la rendrait triviale.
    """
    return df["high"].shift(1).rolling(length).max().rename(f"hh{length}")


def rolling_low(df: pd.DataFrame, length: int) -> pd.Series:
    """Plus bas des `length` barres precedentes (barre courante exclue)."""
    return df["low"].shift(1).rolling(length).min().rename(f"ll{length}")


def realized_volatility(df: pd.DataFrame, length: int = 20) -> pd.Series:
    """Ecart-type glissant des rendements log."""
    returns = np.log(df["close"]).diff()
    out: pd.Series = returns.rolling(length).std(ddof=1).rename(f"volr{length}")
    return out


def atr_ratio(df: pd.DataFrame, fast: int = 5, slow: int = 50) -> pd.Series:
    """Rapport ATR court / ATR long : < 1 = compression, > 1 = expansion.

    Base de la famille "compression -> expansion".
    """
    num = atr(df, fast)
    den = atr(df, slow).replace(0.0, np.nan)
    return (num / den).rename(f"atr_ratio_{fast}_{slow}")


def body_and_wicks(df: pd.DataFrame) -> pd.DataFrame:
    """Anatomie de la bougie, normalisee par son propre range.

    Les fractions permettent de comparer des bougies d'amplitudes tres differentes,
    ce que les valeurs brutes ne permettent pas.
    """
    rng = (df["high"] - df["low"]).replace(0.0, np.nan)
    body = (df["close"] - df["open"]).abs()
    upper = df["high"] - df[["open", "close"]].max(axis=1)
    lower = df[["open", "close"]].min(axis=1) - df["low"]
    return pd.DataFrame(
        {
            "body": body,
            "upper_wick": upper,
            "lower_wick": lower,
            "body_frac": body / rng,
            "upper_wick_frac": upper / rng,
            "lower_wick_frac": lower / rng,
            "bullish": df["close"] > df["open"],
        }
    )


def anchored_vwap(df: pd.DataFrame, anchor: pd.Series) -> pd.Series:
    """VWAP cumulatif reinitialise a chaque changement de `anchor`.

    `anchor` est typiquement une cle de jour ou de seance. En l'absence de volume
    exploitable (les CFD Dukascopy rapportent un volume indicatif), le prix typique
    est pondere par le volume s'il est strictement positif, sinon uniformement — et
    ce repli est explicite plutot que silencieux.
    """
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    volume = df["volume"] if "volume" in df.columns else pd.Series(1.0, index=df.index)
    weight = volume.where(volume > 0, 1.0)
    grouped = weight.groupby(anchor)
    num = (typical * weight).groupby(anchor).cumsum()
    den = grouped.cumsum().replace(0.0, np.nan)
    return (num / den).rename("vwap")


def zscore(series: pd.Series, length: int) -> pd.Series:
    """Ecart a la moyenne glissante, en ecarts-types glissants."""
    mean = series.rolling(length).mean()
    std = series.rolling(length).std(ddof=1).replace(0.0, np.nan)
    return ((series - mean) / std).rename(f"z{length}")
