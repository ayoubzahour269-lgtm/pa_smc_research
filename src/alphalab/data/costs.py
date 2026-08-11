"""Cout d'execution reel, mesure et non suppose.

Le spread est pris sur l'OPEN parce que le moteur entre a l'ouverture de la barre
suivante : c'est le prix que l'on paie reellement, pas une moyenne de la barre.

Ce module existe parce que la seule strategie qui avait semble vivante dans
l'historique de ce projet (creneau 21h->23h, +0.06R au forfait) s'est revelee
perdante a -0.21R une fois le spread reel applique : le pic de rollover de 22h la
mangeait entierement. Un cout forfaitaire est un generateur de faux positifs.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from alphalab.data import registry, snapshot
from alphalab.types import FloatArray


@dataclass(frozen=True, slots=True)
class CostModel:
    """Cout aller-retour en unites de prix, barre par barre.

    `spread` est indexe comme le snapshot BID du couple (symbole, timeframe).
    `source` documente la provenance pour que les rapports ne mentent pas sur ce
    qu'ils mesurent.
    """

    symbol: str
    timeframe: str
    spread: pd.Series
    source: str
    stress_mult: float = 1.0

    def aligned_to(self, index: pd.DatetimeIndex) -> FloatArray:
        """Couts alignes sur un index de travail, en tableau numpy.

        Les barres sans mesure ressortent en NaN : c'est au moteur de decider du
        repli, explicitement, plutot qu'a cette couche de le masquer.
        """
        return (self.spread.reindex(index) * self.stress_mult).to_numpy(dtype=float)

    def stressed(self, mult: float) -> CostModel:
        """Meme modele avec un cout multiplie (test de sensibilite du protocole)."""
        return replace(self, stress_mult=self.stress_mult * mult)

    def describe(self) -> dict[str, object]:
        s = self.spread.dropna() * self.stress_mult
        if s.empty:
            return {
                "symbole": self.symbol,
                "timeframe": self.timeframe,
                "source": self.source,
                "n": 0,
            }
        return {
            "symbole": self.symbol,
            "timeframe": self.timeframe,
            "source": self.source,
            "n": int(s.size),
            "negatifs": int((s < 0).sum()),
            "mediane": round(float(s.median()), 6),
            "moyenne": round(float(s.mean()), 6),
            "p95": round(float(s.quantile(0.95)), 6),
            "p99": round(float(s.quantile(0.99)), 6),
            "max": round(float(s.max()), 6),
        }


class CostUnavailableError(RuntimeError):
    """Le spread reel n'est pas mesurable pour ce couple (symbole, timeframe)."""


def real_spread(symbol: str, timeframe: str) -> pd.Series:
    """`ASK.open - BID.open` barre par barre, aligne sur la grille BID.

    Leve `CostUnavailableError` si l'un des deux cotes manque : mieux vaut un echec
    net qu'un backtest silencieusement fonde sur un cout invente.
    """
    avail = registry.availability(symbol, timeframe)
    if not avail.has_real_spread:
        missing = "ASK" if avail.has_bid else "BID et/ou ASK"
        raise CostUnavailableError(
            f"Spread reel indisponible pour {symbol} {timeframe} : {missing} manquant. "
            f"Telechargez les deux cotes : alphalab freeze {symbol} --tf {timeframe} "
            f"--side BID,ASK"
        )
    assert avail.bid is not None and avail.ask is not None  # garanti par has_real_spread
    bid = snapshot.load(avail.bid, copy=False)
    ask = snapshot.load(avail.ask, copy=False)

    if not bid.index.equals(ask.index):
        common = bid.index.intersection(ask.index)
        if len(common) == 0:
            raise CostUnavailableError(
                f"{symbol} {timeframe} : grilles BID et ASK disjointes, spread incalculable"
            )
        bid, ask = bid.loc[common], ask.loc[common]

    spread = (ask["open"] - bid["open"]).rename("spread")
    return spread


def cost_model(symbol: str, timeframe: str) -> CostModel:
    """Modele de cout fonde sur le spread reel mesure."""
    spread = real_spread(symbol, timeframe)
    return CostModel(
        symbol=symbol,
        timeframe=timeframe,
        spread=spread,
        source="spread reel ASK.open - BID.open",
    )


def hourly_profile(spread: pd.Series) -> pd.DataFrame:
    """Profil du spread par heure UTC.

    Sert a reperer les fenetres structurellement cheres (rollover) avant qu'elles ne
    se deguisent en signal. Sur l'or, la mediane triple a 22h UTC.
    """
    s = spread.dropna()
    grouped = s.groupby(pd.DatetimeIndex(s.index).hour)
    out = pd.DataFrame(
        {
            "n": grouped.size(),
            "mediane": grouped.median(),
            "moyenne": grouped.mean(),
            "p95": grouped.quantile(0.95),
        }
    )
    out.index.name = "heure_utc"
    med = float(s.median())
    out["x_mediane_globale"] = (out["mediane"] / med).round(2) if med else np.nan
    return out
