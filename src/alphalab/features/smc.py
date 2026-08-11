"""Structure de prix (Smart Money Concepts) — definitions mecaniques et causales.

Le depot ne contenait aucun de ces concepts : ni order block, ni FVG, ni sweep, ni
CHoCH. Ils sont ici definis de facon entierement mecanique, parametree et testable.
Aucune definition ne fait appel a l'appreciation d'un operateur.

LE PIEGE CENTRAL — la confirmation retardee. Un pivot de largeur `W` n'est identifiable
que `W` barres apres son sommet. La quasi-totalite des backtests SMC publies exposent le
pivot a sa date de formation : la strategie "sait" alors qu'un sommet est un sommet
avant que le marche ne l'ait confirme, ce qui suffit a fabriquer une courbe d'equite
magnifique et entierement fausse.

Ce module n'expose donc QUE des series causales : la valeur en `t` n'utilise que des
pivots deja confirmes en `t`. La detection brute, non causale, reste privee.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd

from alphalab.features.indicators import atr

DEFAULT_SWING_WIDTH: Final[int] = 3
DEFAULT_IMPULSE_BARS: Final[int] = 3
DEFAULT_IMPULSE_ATR: Final[float] = 1.5
DEFAULT_FVG_MIN_ATR: Final[float] = 0.25
OTE_LOW: Final[float] = 0.618
OTE_HIGH: Final[float] = 0.79


# --------------------------------------------------------------------------------------
# Pivots
# --------------------------------------------------------------------------------------


def _raw_pivots(df: pd.DataFrame, width: int) -> tuple[pd.Series, pd.Series]:
    """Detection brute des pivots — NON CAUSALE, usage interne uniquement.

    Un pivot haut en `k` est un plus-haut strict de la fenetre centree [k-W, k+W].
    Cette fonction regarde donc `W` barres dans le futur : elle ne doit jamais etre
    exposee telle quelle.
    """
    span = 2 * width + 1
    high, low = df["high"], df["low"]
    centered_max = high.rolling(span, center=True).max()
    centered_min = low.rolling(span, center=True).min()
    # `==` sur le max centre : en cas d'egalite parfaite, plusieurs barres peuvent
    # etre marquees. C'est sans consequence, la plus recente l'emporte a l'usage.
    return (high == centered_max).fillna(False), (low == centered_min).fillna(False)


def swing_levels(df: pd.DataFrame, width: int = DEFAULT_SWING_WIDTH) -> pd.DataFrame:
    """Derniers pivots CONFIRMES, disponibles a la bonne date.

    Colonnes :
      - `swing_high` / `swing_low` : prix du dernier pivot confirme ;
      - `swing_high_age` / `swing_low_age` : nombre de barres depuis sa confirmation.

    La confirmation intervient `width` barres apres le pivot, d'ou le `shift(width)`.
    C'est la seule ligne de ce module qui protege de l'anticipation, et elle est
    verifiee par `tests/antilookahead/`.
    """
    is_ph, is_pl = _raw_pivots(df, width)
    high_at_pivot = df["high"].where(is_ph)
    low_at_pivot = df["low"].where(is_pl)

    confirmed_high = high_at_pivot.shift(width).ffill()
    confirmed_low = low_at_pivot.shift(width).ffill()

    positions = pd.Series(np.arange(len(df), dtype=float), index=df.index)
    last_high_pos = positions.where(high_at_pivot.shift(width).notna()).ffill()
    last_low_pos = positions.where(low_at_pivot.shift(width).notna()).ffill()

    return pd.DataFrame(
        {
            "swing_high": confirmed_high,
            "swing_low": confirmed_low,
            "swing_high_age": positions - last_high_pos,
            "swing_low_age": positions - last_low_pos,
        }
    )


# --------------------------------------------------------------------------------------
# Structure : BOS et CHoCH
# --------------------------------------------------------------------------------------


def structure(df: pd.DataFrame, width: int = DEFAULT_SWING_WIDTH) -> pd.DataFrame:
    """Etat de structure, cassures de structure (BOS) et changements de caractere (CHoCH).

    Regles :
      - la structure passe haussiere quand une cloture depasse le dernier pivot haut
        confirme, baissiere quand une cloture passe sous le dernier pivot bas confirme ;
      - **BOS** : cassure dans le sens de la structure deja etablie (continuation) ;
      - **CHoCH** : premiere cassure contraire a la structure en cours (retournement).

    La distinction compte : un BOS et un CHoCH ont la meme signature de prix mais un
    sens opposé, et les confondre revient a melanger continuation et retournement dans
    un meme signal.
    """
    levels = swing_levels(df, width)
    close = df["close"]
    broke_up = (close > levels["swing_high"]).to_numpy()
    broke_down = (close < levels["swing_low"]).to_numpy()

    n = len(df)
    state = np.zeros(n, dtype=np.int8)
    bos = np.zeros(n, dtype=bool)
    choch = np.zeros(n, dtype=bool)

    current = 0
    prev_high = levels["swing_high"].to_numpy()
    prev_low = levels["swing_low"].to_numpy()
    last_broken_high = np.nan
    last_broken_low = np.nan

    for i in range(n):
        if broke_up[i] and prev_high[i] != last_broken_high:
            if current == 1:
                bos[i] = True
            elif current == -1:
                choch[i] = True
            current = 1
            last_broken_high = prev_high[i]
        elif broke_down[i] and prev_low[i] != last_broken_low:
            if current == -1:
                bos[i] = True
            elif current == 1:
                choch[i] = True
            current = -1
            last_broken_low = prev_low[i]
        state[i] = current

    return pd.DataFrame(
        {
            "structure": state,
            "bos": bos,
            "choch": choch,
            "swing_high": levels["swing_high"],
            "swing_low": levels["swing_low"],
        },
        index=df.index,
    )


# --------------------------------------------------------------------------------------
# Premium / discount / OTE
# --------------------------------------------------------------------------------------


def premium_discount(df: pd.DataFrame, width: int = DEFAULT_SWING_WIDTH) -> pd.DataFrame:
    """Position du prix dans la derniere jambe confirmee.

    `pd_position` vaut 0 au pivot bas, 1 au pivot haut. Sous 0,5 le prix est en
    "discount" (zone d'achat pour un biais haussier), au-dessus en "premium".
    La zone OTE (0,618-0,79 de retracement) est exprimee dans le meme repere.
    """
    levels = swing_levels(df, width)
    high, low = levels["swing_high"], levels["swing_low"]
    span = (high - low).replace(0.0, np.nan)
    position = (df["close"] - low) / span
    equilibrium = (high + low) / 2.0
    return pd.DataFrame(
        {
            "equilibrium": equilibrium,
            "pd_position": position,
            "in_discount": position < 0.5,
            "in_premium": position > 0.5,
            # OTE haussiere : retracement de 61,8 a 79 % depuis le haut, donc une
            # position comprise entre 0,21 et 0,382 dans le repere bas->haut.
            "ote_long": position.between(1 - OTE_HIGH, 1 - OTE_LOW),
            "ote_short": position.between(OTE_LOW, OTE_HIGH),
        }
    )


# --------------------------------------------------------------------------------------
# Fair value gaps (imbalances)
# --------------------------------------------------------------------------------------


def fair_value_gaps(
    df: pd.DataFrame, *, min_atr: float = DEFAULT_FVG_MIN_ATR, atr_len: int = 14
) -> pd.DataFrame:
    """Ecarts de valeur sur trois barres, detectes a la cloture de la troisieme.

    FVG haussier : `low[i] > high[i-2]` — le marche a saute une zone de prix en
    montant. FVG baissier : `high[i] < low[i-2]`.

    Le seuil `min_atr` elimine les micro-ecarts qui ne sont que du bruit de cotation :
    sans lui, un instrument peu liquide produit des milliers de "FVG" sans contenu.

    Causal : la barre `i` utilise `i` et `i-2`, toutes deux connues en `i`.
    """
    a = atr(df, atr_len)
    high_2 = df["high"].shift(2)
    low_2 = df["low"].shift(2)

    bull_size = df["low"] - high_2
    bear_size = low_2 - df["high"]
    threshold = a * min_atr

    bull = bull_size > threshold
    bear = bear_size > threshold
    return pd.DataFrame(
        {
            "fvg_bull": bull.fillna(False),
            "fvg_bear": bear.fillna(False),
            "fvg_bull_top": df["low"].where(bull),
            "fvg_bull_bottom": high_2.where(bull),
            "fvg_bear_top": low_2.where(bear),
            "fvg_bear_bottom": df["high"].where(bear),
            "fvg_size_atr": (bull_size.where(bull, bear_size.where(bear)) / a),
        }
    )


# --------------------------------------------------------------------------------------
# Order blocks
# --------------------------------------------------------------------------------------


def order_blocks(
    df: pd.DataFrame,
    *,
    width: int = DEFAULT_SWING_WIDTH,
    impulse_bars: int = DEFAULT_IMPULSE_BARS,
    impulse_atr: float = DEFAULT_IMPULSE_ATR,
    atr_len: int = 14,
) -> pd.DataFrame:
    """Derniere bougie de sens oppose precedant un mouvement impulsif casseur.

    Definition retenue, entierement mecanique :
      1. une cassure de structure (BOS ou CHoCH) survient en `i` ;
      2. le deplacement sur les `impulse_bars` dernieres barres vaut au moins
         `impulse_atr` x ATR — c'est le critere d'impulsivite, sans lequel n'importe
         quelle derive lente produirait des order blocks ;
      3. l'order block est la derniere bougie de couleur opposee dans cette fenetre.

    Les colonnes rendues decrivent le dernier order block VALIDE connu en `t`. Causal :
    la recherche ne regarde que vers le passe depuis une cassure deja survenue.
    """
    struct = structure(df, width)
    a = atr(df, atr_len).to_numpy()
    o = df["open"].to_numpy()
    h = df["high"].to_numpy()
    low = df["low"].to_numpy()
    c = df["close"].to_numpy()
    n = len(df)

    broke = (struct["bos"] | struct["choch"]).to_numpy()
    direction = struct["structure"].to_numpy()

    ob_top = np.full(n, np.nan)
    ob_bottom = np.full(n, np.nan)
    ob_dir = np.zeros(n, dtype=np.int8)
    ob_age = np.full(n, np.nan)

    cur_top = cur_bottom = np.nan
    cur_dir = 0
    cur_pos = -1

    for i in range(n):
        if broke[i] and np.isfinite(a[i]) and a[i] > 0:
            start = max(i - impulse_bars, 0)
            move = abs(c[i] - o[start])
            if move >= impulse_atr * a[i]:
                d = int(direction[i])
                found = -1
                for j in range(i, start - 1, -1):
                    bullish_candle = c[j] > o[j]
                    if (d == 1 and not bullish_candle) or (d == -1 and bullish_candle):
                        found = j
                        break
                if found >= 0:
                    cur_top, cur_bottom = h[found], low[found]
                    cur_dir, cur_pos = d, found
        ob_top[i], ob_bottom[i], ob_dir[i] = cur_top, cur_bottom, cur_dir
        ob_age[i] = (i - cur_pos) if cur_pos >= 0 else np.nan

    price_in = (df["low"].to_numpy() <= ob_top) & (df["high"].to_numpy() >= ob_bottom)
    return pd.DataFrame(
        {
            "ob_top": ob_top,
            "ob_bottom": ob_bottom,
            "ob_direction": ob_dir,
            "ob_age": ob_age,
            "ob_mitigated": price_in & (ob_dir != 0),
        },
        index=df.index,
    )


# --------------------------------------------------------------------------------------
# Liquidite : egalites, balayages, chasses aux stops
# --------------------------------------------------------------------------------------


def equal_levels(
    df: pd.DataFrame, *, lookback: int = 20, tolerance_atr: float = 0.1, atr_len: int = 14
) -> pd.DataFrame:
    """Plus-hauts / plus-bas quasi identiques dans les `lookback` barres precedentes.

    Ce sont les zones ou s'accumulent les ordres stop, donc les cibles naturelles d'un
    balayage. La tolerance est exprimee en ATR pour rester comparable entre instruments.
    """
    a = atr(df, atr_len)
    prior_high = df["high"].shift(1).rolling(lookback).max()
    prior_low = df["low"].shift(1).rolling(lookback).min()
    tol = a * tolerance_atr
    return pd.DataFrame(
        {
            "eq_high": prior_high,
            "eq_low": prior_low,
            "near_eq_high": (df["high"] - prior_high).abs() <= tol,
            "near_eq_low": (df["low"] - prior_low).abs() <= tol,
        }
    )


def liquidity_sweeps(
    df: pd.DataFrame,
    reference_high: pd.Series,
    reference_low: pd.Series,
    *,
    min_wick_atr: float = 0.1,
    atr_len: int = 14,
) -> pd.DataFrame:
    """Balayage : la meche traverse un niveau de reference, la cloture revient en deca.

    C'est la signature mecanique de la "chasse aux stops" : le prix va chercher la
    liquidite au-dela d'un extreme connu, puis la rejette dans la meme barre. Une
    simple cassure suivie d'une cloture au-dela n'est PAS un balayage — c'est une
    cassure, et le module les distingue.

    `reference_high` / `reference_low` sont fournis par l'appelant (extremes de seance,
    plus-haut de la veille, egalites) et doivent eux-memes etre causaux.
    """
    a = atr(df, atr_len)
    min_wick = a * min_wick_atr

    pierced_up = df["high"] > reference_high
    rejected_up = df["close"] < reference_high
    wick_up = df["high"] - reference_high

    pierced_down = df["low"] < reference_low
    rejected_down = df["close"] > reference_low
    wick_down = reference_low - df["low"]

    return pd.DataFrame(
        {
            "sweep_high": (pierced_up & rejected_up & (wick_up >= min_wick)).fillna(False),
            "sweep_low": (pierced_down & rejected_down & (wick_down >= min_wick)).fillna(False),
            "sweep_high_depth_atr": (wick_up / a).where(pierced_up),
            "sweep_low_depth_atr": (wick_down / a).where(pierced_down),
            # Cassure franche : traverse ET cloture au-dela. L'oppose d'un balayage.
            "breakout_high": (pierced_up & ~rejected_up).fillna(False),
            "breakout_low": (pierced_down & ~rejected_down).fillna(False),
        }
    )


def build_all(
    df: pd.DataFrame,
    *,
    width: int = DEFAULT_SWING_WIDTH,
    atr_len: int = 14,
) -> pd.DataFrame:
    """Assemble toutes les features SMC en une trame unique.

    Point d'entree utilise par les familles d'alpha et par les tests d'anti-anticipation
    (qui rejouent CE point d'entree, garantissant qu'aucune feature n'echappe au controle).
    """
    levels = equal_levels(df, atr_len=atr_len)
    parts = [
        structure(df, width),
        premium_discount(df, width),
        fair_value_gaps(df, atr_len=atr_len),
        order_blocks(df, width=width, atr_len=atr_len),
        levels,
        liquidity_sweeps(df, levels["eq_high"], levels["eq_low"], atr_len=atr_len),
    ]
    out = pd.concat(parts, axis=1)
    return out.loc[:, ~out.columns.duplicated()]
