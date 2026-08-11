"""Seances de marche, killzones et ranges de seance completes.

Toutes les heures sont en UTC. Les bornes sont approximatives par nature (les seances
se chevauchent, l'heure d'ete deplace les ouvertures cash d'une heure) : elles sont
donc parametrees et pre-enregistrees, jamais devinees au moment de l'analyse.

Point de vigilance central : le range d'une seance n'est connu qu'une fois la seance
TERMINEE. Une feature qui expose le haut de la seance asiatique a 03h du matin est une
anticipation — et c'est l'une des plus frequentes dans la litterature price action.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

#: Heure UTC a laquelle bascule la journee de trading. 22h correspond au rollover des
#: courtiers FX/CFD — c'est aussi le pic de spread observe sur l'or (3,2x la mediane).
DEFAULT_ROLL_HOUR: Final[int] = 22


@dataclass(frozen=True, slots=True)
class SessionWindow:
    """Fenetre horaire d'une seance, bornes en heures UTC, fin exclue."""

    name: str
    start_hour: int
    end_hour: int

    def mask(self, index: pd.DatetimeIndex) -> pd.Series:
        hour = pd.Series(index.hour, index=index)
        if self.start_hour < self.end_hour:
            inside = (hour >= self.start_hour) & (hour < self.end_hour)
        else:  # fenetre qui franchit minuit
            inside = (hour >= self.start_hour) | (hour < self.end_hour)
        return inside.rename(f"in_{self.name}")


#: Seances principales. Bornes volontairement larges et non chevauchantes pour servir
#: de partition ; les killzones ci-dessous sont les sous-fenetres etroites.
SESSIONS: Final[tuple[SessionWindow, ...]] = (
    SessionWindow("asie", 0, 7),
    SessionWindow("londres", 7, 13),
    SessionWindow("newyork", 13, 21),
    SessionWindow("rollover", 21, 24),
)

#: Sous-fenetres a forte activite, telles que definies par la litterature price action.
KILLZONES: Final[tuple[SessionWindow, ...]] = (
    SessionWindow("kz_londres", 7, 10),
    SessionWindow("kz_newyork", 12, 15),
    SessionWindow("kz_cloture_ny", 19, 21),
)


def trading_day(index: pd.DatetimeIndex, roll_hour: int = DEFAULT_ROLL_HOUR) -> pd.Series:
    """Cle de journee de trading, basculant a `roll_hour` UTC.

    Utiliser la journee calendaire UTC couperait la seance de New York en deux, ce qui
    fausse tout ancrage (VWAP, range du jour, gap d'ouverture).
    """
    shifted = index - pd.Timedelta(hours=roll_hour)
    return pd.Series(shifted.normalize(), index=index, name="trading_day")


def session_label(index: pd.DatetimeIndex) -> pd.Series:
    """Nom de la seance en cours pour chaque barre."""
    out = pd.Series("inconnue", index=index, name="session", dtype=object)
    for window in SESSIONS:
        out = out.mask(window.mask(index), window.name)
    return out


def session_masks(index: pd.DatetimeIndex, *, killzones: bool = True) -> pd.DataFrame:
    """Colonnes booleennes, une par seance (et par killzone si demande)."""
    windows = (*SESSIONS, *KILLZONES) if killzones else SESSIONS
    return pd.DataFrame({w.mask(index).name: w.mask(index) for w in windows})


def completed_window_range(
    df: pd.DataFrame,
    window: SessionWindow,
    *,
    roll_hour: int = DEFAULT_ROLL_HOUR,  # noqa: ARG001 - garde la signature stable
) -> pd.DataFrame:
    """Haut, bas et milieu de la DERNIERE occurrence TERMINEE de la fenetre.

    La valeur en `t` provient de la derniere occurrence dont la barre de cloture est
    strictement anterieure a `t`. Tant que la premiere occurrence n'est pas close, les
    colonnes valent NaN — c'est voulu : il n'y a rien a savoir avant.

    Les occurrences sont identifiees comme des sequences CONTIGUES de barres dans la
    fenetre, et non par journee de trading. La nuance est necessaire : une fenetre qui
    franchit la bascule de journee (le rollover 21h-24h face a une bascule a 22h)
    formerait sinon un groupe discontinu, dont la "cloture" tomberait le lendemain — et
    la feature laisserait alors fuiter le futur. Ce cas precis a ete detecte par
    `tests/antilookahead/`, pas par relecture.
    """
    index = pd.DatetimeIndex(df.index)
    inside = window.mask(index)
    empty = pd.Series(np.nan, index=index)
    if not bool(inside.any()):
        return pd.DataFrame(
            {
                f"{window.name}_high": empty,
                f"{window.name}_low": empty,
                f"{window.name}_mid": empty,
            }
        )

    mask = inside.to_numpy()
    occurrence = np.cumsum(mask & ~np.concatenate([[False], mask[:-1]]))
    sub = df.loc[mask]
    sub_occ = pd.Series(occurrence[mask], index=sub.index)

    highs = sub["high"].groupby(sub_occ).max()
    lows = sub["low"].groupby(sub_occ).min()
    closes_at = pd.Series(sub.index, index=sub.index).groupby(sub_occ).max()

    known = pd.DataFrame(
        {"high": highs.to_numpy(), "low": lows.to_numpy()},
        index=pd.DatetimeIndex(closes_at.to_numpy()),
    ).sort_index()
    # Une barre situee A la cloture de l'occurrence ne peut pas encore l'exploiter :
    # le decalage d'un nanoseconde rend la disponibilite strictement posterieure.
    known.index = known.index + pd.Timedelta(nanoseconds=1)
    known = known[~known.index.duplicated(keep="last")]

    aligned = known.reindex(known.index.union(index)).ffill().reindex(index)
    return pd.DataFrame(
        {
            f"{window.name}_high": aligned["high"].to_numpy(),
            f"{window.name}_low": aligned["low"].to_numpy(),
            f"{window.name}_mid": ((aligned["high"] + aligned["low"]) / 2.0).to_numpy(),
        },
        index=index,
    )


def previous_day_levels(df: pd.DataFrame, *, roll_hour: int = DEFAULT_ROLL_HOUR) -> pd.DataFrame:
    """Haut, bas et cloture de la journee de trading PRECEDENTE.

    Niveaux de reference universels : ce sont eux que la plupart des sweeps de
    liquidite viennent chercher.
    """
    index = pd.DatetimeIndex(df.index)
    day = trading_day(index, roll_hour)
    daily = df.groupby(day).agg(high=("high", "max"), low=("low", "min"), close=("close", "last"))
    prev = daily.shift(1)
    prev.columns = pd.Index(["pdh", "pdl", "pdc"])
    joined = day.to_frame().join(prev, on="trading_day")
    return joined[["pdh", "pdl", "pdc"]].set_axis(index)


def session_progress(df: pd.DataFrame, *, roll_hour: int = DEFAULT_ROLL_HOUR) -> pd.Series:
    """Fraction ecoulee de la journee de trading, entre 0 et 1.

    Mesure le TEMPS ecoule, pas le rang de la barre. La nuance est essentielle : le
    rang normalise par le nombre de barres du jour exigerait de connaitre ce nombre a
    l'avance, ce qui est une anticipation. Le temps ecoule, lui, ne depend que de `t`.
    """
    index = pd.DatetimeIndex(df.index)
    shifted = index - pd.Timedelta(hours=roll_hour)
    elapsed = (shifted - shifted.normalize()).total_seconds() / 86_400.0
    return pd.Series(elapsed, index=index, name="session_progress")
