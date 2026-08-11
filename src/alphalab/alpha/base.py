"""Interface commune des familles d'hypotheses.

Une famille repond a UNE question : "a quel moment un setup est-il present, et dans
quel sens ?". Elle ne decide pas de trader, ne choisit pas de taille, et ne juge pas de
sa propre rentabilite. Ces trois decisions appartiennent respectivement a la
calibration, au moteur de risque et au protocole.

Cette separation est ce qui permet de comparer sur une meme echelle des approches aussi
heterogenes qu'un effet de seance et une structure SMC : le moteur ne voit d'elles que
des `Order` identiquement formes.

Ajouter une piste d'exploration coute donc un fichier dans `families/`, pas une
refonte. C'est l'inverse de la version precedente du projet, ou chaque hypothese etait
codee a la main dans un script jetable — d'ou une exploration arretee a huit essais.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np
import pandas as pd

from alphalab.backtest.engine import FIXED_EXIT, ExitPolicy, Order
from alphalab.data.costs import CostModel
from alphalab.features import indicators


@dataclass(frozen=True, slots=True)
class Geometry:
    """Geometrie de risque, en multiples d'ATR — PRE-ENREGISTREE.

    Le choix du stop et de l'objectif pese autant sur le resultat que le signal
    lui-meme. Le figer avant l'exploration evite de le regler a posteriori sur ce qui
    marche, ce qui produirait un edge purement retrospectif.
    """

    sl_atr: float = 1.0
    tp_atr: float = 2.0
    max_hold: int = 32
    atr_len: int = 14

    @property
    def label(self) -> str:
        return f"sl{self.sl_atr}xtp{self.tp_atr}xh{self.max_hold}"


#: Geometries gelees, une par timeframe d'execution. Toute variation compte comme un
#: essai supplementaire au sens de la correction pour tests multiples.
GEOMETRY: Final[dict[str, Geometry]] = {
    "M15": Geometry(sl_atr=1.0, tp_atr=2.0, max_hold=32, atr_len=14),  # ~8 h de detention
    "H1": Geometry(sl_atr=1.0, tp_atr=2.0, max_hold=24, atr_len=14),  # ~1 journee
}


@dataclass(slots=True)
class Context:
    """Tout ce qu'une famille peut consulter pour un symbole donne.

    Les features partagees sont calculees une seule fois et passees a toutes les
    familles : c'est ce qui rend l'exploration large abordable en temps de calcul, et
    ce qui garantit que deux familles parlent bien du meme ATR.
    """

    symbol: str
    timeframe: str
    df: pd.DataFrame
    features: pd.DataFrame
    cost: CostModel
    geometry: Geometry
    peers: dict[str, pd.DataFrame] = field(default_factory=dict)

    @property
    def atr(self) -> pd.Series:
        return self.features["atr"]

    def geometry_at(self, position: int) -> tuple[float, float] | None:
        """Stop et objectif en unites de prix a une position donnee.

        Rend `None` quand l'ATR n'est pas encore defini (periode de chauffe) ou nul :
        un stop nul rendrait le R infini, ce que le moteur refuse a juste titre.
        """
        a = float(self.atr.to_numpy()[position])
        if not np.isfinite(a) or a <= 0:
            return None
        return (self.geometry.sl_atr * a, self.geometry.tp_atr * a)

    def orders_from_signal(
        self, signal: pd.Series, tag: str, exit_policy: ExitPolicy = FIXED_EXIT
    ) -> list[Order]:
        """Convertit une serie de signaux (-1 / 0 / +1) en ordres executables.

        Point de vigilance : le signal en `t` declenche une entree en `t+1`. La serie
        doit donc etre causale — c'est garanti par le contrat de `features`, verifie
        par `tests/antilookahead/`.
        """
        values = signal.reindex(self.df.index).fillna(0).to_numpy()
        atr_values = self.atr.to_numpy()
        orders: list[Order] = []
        for position in np.flatnonzero(values):
            direction = int(np.sign(values[position]))
            a = float(atr_values[position])
            if not np.isfinite(a) or a <= 0:
                continue
            orders.append(
                Order(
                    symbol=self.symbol,
                    signal_ts=self.df.index[position],
                    direction=direction,
                    stop_distance=self.geometry.sl_atr * a,
                    target_distance=self.geometry.tp_atr * a,
                    max_hold=self.geometry.max_hold,
                    tag=tag,
                    exit_policy=exit_policy,
                )
            )
        return orders


class AlphaFamily(ABC):
    """Famille d'hypotheses. Une sous-classe = une piste d'exploration."""

    #: Identifiant court, stable, utilise dans le journal des essais.
    name: str = ""
    #: Ce que la famille teste, en une phrase.
    question: str = ""
    #: Regle de sortie. Par defaut celle des campagnes S6-S9 : stop et objectif fixes.
    #: Une famille qui la change teste une hypothese differente, et compte donc comme
    #: un essai distinct.
    exit_policy: ExitPolicy = FIXED_EXIT
    #: Suffixe d'etiquette quand la famille ne differe que par sa sortie. `None` quand
    #: la sortie est celle de reference et n'a pas a encombrer le nom.
    exit_label: str | None = None

    def parameters(self) -> dict[str, Any]:
        """Parametres effectifs, journalises avant lecture du resultat.

        La politique de sortie y figure toujours : deux familles d'entree identique
        mais de sortie differente testent des hypotheses differentes et doivent compter
        comme deux essais.
        """
        return {"sortie": self.exit_policy.label}

    @abstractmethod
    def signal(self, ctx: Context) -> pd.Series:
        """Serie causale a valeurs dans {-1, 0, +1}, alignee sur `ctx.df.index`."""

    def generate(self, ctx: Context) -> list[Order]:
        """Ordres produits par la famille pour ce contexte."""
        return ctx.orders_from_signal(self.signal(ctx), self.name, self.exit_policy)

    def __repr__(self) -> str:  # pragma: no cover - confort de debogage
        return f"<{type(self).__name__} {self.name}>"


def build_features(df: pd.DataFrame, geometry: Geometry) -> pd.DataFrame:
    """Socle de features partage par toutes les familles.

    Volontairement minimal : chaque famille enrichit ce socle avec ce qui lui est
    propre. Y entasser toutes les features possibles couterait du temps de calcul a
    toutes les familles pour le benefice d'une seule.
    """
    out = pd.DataFrame(index=df.index)
    out["atr"] = indicators.atr(df, geometry.atr_len)
    out["ema50"] = indicators.ema(df["close"], 50)
    out["ema200"] = indicators.ema(df["close"], 200)
    out["rsi14"] = indicators.rsi(df["close"], 14)
    out["atr_ratio"] = indicators.atr_ratio(df)
    out = pd.concat([out, indicators.adx(df, 14), indicators.body_and_wicks(df)], axis=1)
    return out


def clean_signal(raw: pd.Series, index: pd.Index) -> pd.Series:
    """Normalise une serie quelconque en signal -1 / 0 / +1 aligne sur `index`."""
    return (
        pd.Series(raw, index=raw.index)
        .reindex(index)
        .fillna(0)
        .clip(-1, 1)
        .round()
        .astype(int)
        .rename("signal")
    )
