"""Porte de faisabilite du scalping : le cout tient-il dans l'amplitude disponible ?

Cette mesure precede toute strategie. Elle ne demande pas "quel signal prendre" mais
"reste-t-il quelque chose a prendre apres le spread". C'est la question que la plupart
des robots de scalping vendus dans le commerce n'ont jamais posee, et c'est celle qui
decide de tout : un edge de 3 points sur un marche qui coute 2,5 points a l'aller-retour
n'est pas un petit edge, c'est un edge inexistant.

Trois grandeurs, toutes mesurees et aucune supposee :

  1. `spread median / true range median` — la part du mouvement d'une barre qui part en
     frais. Independant de toute strategie.
  2. `spread median / excursion favorable mediane a H barres` — la meme chose rapportee
     a ce qu'un trade peut reellement viser. C'est la grandeur pre-enregistree dans
     `config.SCALPING_MAX_COST_FRAC_OF_MFE`.
  3. Le **taux de reussite minimal** qui en decoule. Pour un trade symetrique qui vise
     `x` et risque `x` en payant `c` :

         p*x - (1-p)*x - c = 0   =>   p = 0.5 + c / (2x)

     Une fraction de cout de 25 % impose donc 62,5 % de reussite pour ne rien gagner.
     C'est la traduction concrete du seuil pre-enregistre, et la raison de son niveau.

**Loi d'echelle.** Le spread est a peu pres constant en unites de prix quand on
descend en timeframe, tandis que l'amplitude croit comme la racine du temps. La
fraction de cout doit donc croitre comme `sqrt(T_ref / T)`. On dispose de M15 et H1 :
le rapport mesure entre les deux permet de VERIFIER cette loi sur nos propres donnees
avant de s'en servir pour extrapoler vers M5 et M1, qui sont absents du disque. Une
extrapolation verifiee sur un point vaut mieux qu'une regle invoquee.

Note sur l'anticipation : ce module regarde volontairement vers le futur (l'excursion
favorable est par definition une grandeur ex post). Ce n'est pas une feature et cela ne
doit jamais entrer dans un signal — c'est un diagnostic sur ce que le marche a offert.
Le contrat causal ne s'y applique pas, et c'est pourquoi il vit dans `backtest/` et non
dans `features/`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

from alphalab.config import HOLDOUT_START, SCALPING_MAX_COST_FRAC_OF_MFE, in_sample_start
from alphalab.data import costs, registry, snapshot
from alphalab.features import indicators

#: Duree d'une barre en minutes, pour la loi d'echelle.
MINUTES: Final[dict[str, int]] = {"M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 60, "H4": 240}

#: Horizons de detention evalues, en barres. Un scalpeur vit dans les premiers.
DEFAULT_HORIZONS: Final[tuple[int, ...]] = (1, 2, 4, 8)


@dataclass(frozen=True, slots=True)
class HorizonMeasure:
    """Faisabilite a un horizon de detention donne."""

    horizon: int
    mfe_median: float
    cost_fraction: float
    breakeven_winrate: float

    @property
    def viable(self) -> bool:
        return self.cost_fraction <= SCALPING_MAX_COST_FRAC_OF_MFE

    def as_row(self) -> dict[str, object]:
        return {
            "horizon_barres": self.horizon,
            "excursion_mediane": round(self.mfe_median, 4),
            "fraction_cout": round(self.cost_fraction, 4),
            "taux_reussite_min": round(self.breakeven_winrate, 4),
            "viable": "oui" if self.viable else "NON",
        }


@dataclass(frozen=True, slots=True)
class FeasibilityReport:
    """Verdict de faisabilite pour un couple (symbole, timeframe)."""

    symbol: str
    timeframe: str
    n_bars: int
    start: pd.Timestamp
    end: pd.Timestamp
    spread_median: float
    true_range_median: float
    horizons: tuple[HorizonMeasure, ...]

    @property
    def cost_over_range(self) -> float:
        """Part du mouvement d'UNE barre absorbee par le spread."""
        if self.true_range_median <= 0:
            return float("nan")
        return self.spread_median / self.true_range_median

    @property
    def first_viable_horizon(self) -> int | None:
        """Plus court horizon ou le cout tient sous le seuil pre-enregistre."""
        for measure in self.horizons:
            if measure.viable:
                return measure.horizon
        return None

    def summary(self) -> dict[str, object]:
        return {
            "symbole": self.symbol,
            "timeframe": self.timeframe,
            "barres": self.n_bars,
            "debut": str(self.start.date()),
            "fin": str(self.end.date()),
            "spread_median": round(self.spread_median, 4),
            "range_median": round(self.true_range_median, 4),
            "cout_sur_range": round(self.cost_over_range, 4),
            "1er_horizon_viable": self.first_viable_horizon or "aucun",
        }


def forward_excursion(df: pd.DataFrame, horizon: int) -> pd.Series:
    """Excursion favorable mediane d'une entree a l'ouverture de la barre suivante.

    Neutre en direction : la moyenne de ce qu'aurait offert un achat et de ce qu'aurait
    offert une vente. Prendre le meilleur des deux reviendrait a mesurer l'amplitude
    d'un trader devin, ce qui gonflerait l'excursion et ferait passer la porte a tort.
    """
    if horizon < 1:
        raise ValueError("horizon doit valoir au moins 1 barre")
    entry = df["open"].shift(-1)
    # `h[::-1].rolling(n).max()[::-1]` en t vaut max(h[t .. t+n-1]) ; le `shift(-1)`
    # decale la fenetre sur [t+1 .. t+horizon], soit exactement la duree de detention.
    high = df["high"][::-1].rolling(horizon, min_periods=horizon).max()[::-1].shift(-1)
    low = df["low"][::-1].rolling(horizon, min_periods=horizon).min()[::-1].shift(-1)
    up = (high - entry).clip(lower=0.0)
    down = (entry - low).clip(lower=0.0)
    return ((up + down) / 2.0).rename("mfe")


def assess(
    symbol: str,
    timeframe: str,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    *,
    in_sample_only: bool = True,
) -> FeasibilityReport:
    """Mesure la faisabilite sur les donnees reellement presentes sur le disque.

    Par defaut la mesure s'arrete au debut du hold-out : meme un diagnostic de cout
    consomme de l'information si on le regarde sur la periode reservee.
    """
    availability = registry.availability(symbol, timeframe)
    if availability.bid is None:
        raise costs.CostUnavailableError(f"{symbol} {timeframe} : snapshot BID absent")
    df = snapshot.load(availability.bid)
    spread = costs.real_spread(symbol, timeframe)

    if in_sample_only:
        window = (df.index >= in_sample_start(symbol)) & (df.index < HOLDOUT_START)
        df = df.loc[window]
        spread = spread.loc[spread.index.intersection(df.index)]

    if df.empty:
        raise costs.CostUnavailableError(f"{symbol} {timeframe} : aucune barre in-sample")

    spread_median = float(spread.dropna().median())
    range_median = float(indicators.true_range(df).dropna().median())

    measures: list[HorizonMeasure] = []
    for horizon in horizons:
        mfe = float(forward_excursion(df, horizon).dropna().median())
        fraction = spread_median / mfe if mfe > 0 else float("inf")
        measures.append(
            HorizonMeasure(
                horizon=horizon,
                mfe_median=mfe,
                cost_fraction=fraction,
                breakeven_winrate=0.5 + fraction / 2.0,
            )
        )

    index = pd.DatetimeIndex(df.index)
    return FeasibilityReport(
        symbol=symbol,
        timeframe=timeframe,
        n_bars=int(len(df)),
        start=index[0],
        end=index[-1],
        spread_median=spread_median,
        true_range_median=range_median,
        horizons=tuple(measures),
    )


def duration_points(reports: dict[str, FeasibilityReport]) -> list[tuple[int, float]]:
    """Fractions de cout mesurees, indexees par DUREE de detention en minutes.

    Le decoupage en barres est une convention d'affichage ; ce qui coute, c'est le temps
    passe en position. Deux grilles differentes qui couvrent la meme duree doivent donc
    donner la meme fraction — c'est verifiable ici, et c'est verifie.
    """
    points: list[tuple[int, float]] = []
    for timeframe, report in reports.items():
        step = MINUTES.get(timeframe)
        if step is None:
            continue
        points.extend(
            (measure.horizon * step, measure.cost_fraction) for measure in report.horizons
        )
    return sorted(points)


def grid_agreement(reports: dict[str, FeasibilityReport]) -> pd.DataFrame:
    """Confrontation des grilles la ou elles couvrent la meme duree.

    C'est le controle qui autorise tout le reste : si M15 sur 4 barres et H1 sur 1 barre
    ne donnaient pas la meme fraction de cout, la grandeur dependrait de la grille et
    l'extrapolation vers M1 n'aurait aucun sens.
    """
    by_duration: dict[int, dict[str, float]] = {}
    for timeframe, report in reports.items():
        step = MINUTES.get(timeframe)
        if step is None:
            continue
        for measure in report.horizons:
            by_duration.setdefault(measure.horizon * step, {})[timeframe] = measure.cost_fraction

    rows: list[dict[str, object]] = []
    for minutes, values in sorted(by_duration.items()):
        if len(values) < 2:
            continue
        lo, hi = min(values.values()), max(values.values())
        rows.append(
            {
                "duree_min": minutes,
                **{f"cout_{tf}": round(v, 4) for tf, v in sorted(values.items())},
                "ecart_relatif": round((hi - lo) / lo, 4) if lo > 0 else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def duration_law(reports: dict[str, FeasibilityReport]) -> float:
    """Constante `k` de la loi `fraction(T) = k / sqrt(T)`, T en minutes.

    Estimee par la mediane de `fraction * sqrt(T)` sur tous les points mesures : robuste
    aux quelques horizons ou l'echantillon est le plus mince.
    """
    products = [
        fraction * float(np.sqrt(minutes)) for minutes, fraction in duration_points(reports)
    ]
    if not products:
        raise ValueError("aucun point mesure : loi de duree inestimable")
    return float(np.median(products))


def min_viable_duration(reports: dict[str, FeasibilityReport]) -> float:
    """Duree de detention minimale, en minutes, pour tenir sous le seuil pre-enregistre.

    `k / sqrt(T) <= seuil  =>  T >= (k / seuil)^2`. C'est la reponse chiffree a la
    question "a partir de quand un trade a-t-il de quoi payer son spread".
    """
    return float((duration_law(reports) / SCALPING_MAX_COST_FRAC_OF_MFE) ** 2)


def duration_table(
    reports: dict[str, FeasibilityReport],
    durations: tuple[int, ...] = (1, 5, 15, 30, 60, 120, 240),
) -> pd.DataFrame:
    """Cout et taux de reussite minimal en fonction de la duree de detention visee.

    Les durees couvertes par une grille presente sont mesurees ; les autres — typiquement
    la minute et les cinq minutes, absentes du disque — sont deduites de la loi ajustee.
    La colonne `nature` empeche de confondre les deux.
    """
    k = duration_law(reports)
    measured = dict(duration_points(reports))
    rows: list[dict[str, object]] = []
    for minutes in durations:
        fraction = measured.get(minutes, k / float(np.sqrt(minutes)))
        rows.append(
            {
                "duree_min": minutes,
                "fraction_cout": round(fraction, 4),
                "taux_reussite_min": round(0.5 + fraction / 2.0, 4),
                "viable": "oui" if fraction <= SCALPING_MAX_COST_FRAC_OF_MFE else "NON",
                "nature": "mesure" if minutes in measured else "extrapolation",
            }
        )
    return pd.DataFrame(rows)


def scaling_law(reports: dict[str, FeasibilityReport]) -> pd.DataFrame:
    """Verifie la loi en racine du temps, puis extrapole vers les timeframes absents.

    L'ancre est le timeframe le plus court REELLEMENT mesure. Les lignes marquees
    `mesure` viennent des donnees ; celles marquees `extrapolation` sont un calcul, et
    le tableau le dit explicitement plutot que de les melanger.
    """
    measured = {tf: rep for tf, rep in reports.items() if tf in MINUTES}
    if not measured:
        return pd.DataFrame()
    anchor_tf = min(measured, key=lambda tf: MINUTES[tf])
    anchor = measured[anchor_tf]

    rows: list[dict[str, object]] = []
    for tf in sorted(MINUTES, key=lambda t: MINUTES[t]):
        ratio = float(np.sqrt(MINUTES[anchor_tf] / MINUTES[tf]))
        predicted = anchor.cost_over_range * ratio
        report = measured.get(tf)
        rows.append(
            {
                "timeframe": tf,
                "minutes": MINUTES[tf],
                "cout_sur_range_predit": round(predicted, 4),
                "cout_sur_range_mesure": (
                    round(report.cost_over_range, 4) if report is not None else ""
                ),
                "nature": "mesure" if report is not None else "extrapolation",
            }
        )
    return pd.DataFrame(rows)
