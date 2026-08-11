"""Exploration systematique : toutes les familles, un seul protocole, un seul verdict.

Deroulement, dans cet ordre strict :

  1. chargement des donnees in-sample (le hold-out reste ferme) ;
  2. couche de cout reel, par symbole ;
  3. pour chaque famille et chaque symbole : **enregistrement de l'essai au journal**,
     puis generation des candidats, puis passage du gate ;
  4. correction pour tests multiples sur l'ENSEMBLE des essais journalises ;
  5. verdict.

L'ordre du point 3 n'est pas cosmetique. Enregistrer l'essai avant d'en lire le
resultat est la seule facon d'obtenir un compte d'essais honnete : a posteriori, on
oublie systematiquement ceux qui n'ont rien donne, et la correction devient decorative.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from alphalab.alpha import diversity
from alphalab.alpha.base import GEOMETRY, Context, build_features
from alphalab.alpha.registry import all_families, family_label
from alphalab.backtest import protocol
from alphalab.backtest.engine import ExecConfig, Order, SymbolData
from alphalab.config import (
    HOLDOUT_START,
    N_BOOTSTRAP,
    N_RANDOM_CONTROL,
    SEED,
    in_sample_start,
)
from alphalab.data import costs, registry, snapshot
from alphalab.model.multipletesting import MultipleTestResult, benjamini_hochberg


@dataclass(frozen=True, slots=True)
class ExplorationResult:
    """Sortie complete d'une campagne d'exploration."""

    table: pd.DataFrame
    correction: MultipleTestResult
    timeframe: str
    symbols: tuple[str, ...]
    missing_peers: tuple[str, ...]
    #: Correction alternative, calculee sur le nombre EFFECTIF de tests independants.
    #: Publiee comme borne basse informative, jamais comme critere de retenue.
    correction_effective: MultipleTestResult | None = None
    diversity_report: diversity.DiversityReport | None = None

    @property
    def survivors(self) -> pd.DataFrame:
        """Familles franchissant le gate ET resistant a la correction multiple."""
        if self.table.empty:
            return self.table
        return self.table[self.table["survit_correction"]]

    def verdict(self) -> str:
        """Verdict combine.

        Attention a ne pas confondre deux nombres : la correction de Benjamini-Hochberg
        peut retenir une hypothese dont l'esperance-R est NEGATIVE — elle ne juge que
        la p-value du temoin. Seule la conjonction "franchit les portes du protocole ET
        resiste a la correction" constitue un resultat. C'est ce nombre-la qui est
        annonce en premier ; la sortie brute de la correction n'arrive qu'ensuite, comme
        detail technique.
        """
        n = len(self.survivors)
        if n == 0:
            lines = [
                f"AUCUNE famille retenue sur {len(self.table)} evaluees "
                f"({self.correction.n_trials} essais journalises, FDR = {self.correction.alpha})."
            ]
        else:
            noms = ", ".join(
                f"{r.famille}@{r.symbole}" for r in self.survivors.itertuples()
            )
            lines = [
                f"{n} famille(s) retenue(s) sur {len(self.table)} evaluees : {noms}. "
                "Resultat in-sample uniquement — le hold-out reste seul juge."
            ]
        lines.append(f"Detail de la correction seule : {self.correction.summary()}")
        if self.diversity_report is not None:
            lines.append(self.diversity_report.summary())
        if self.correction_effective is not None:
            lines.append(
                "Borne basse (correction sur les tests EFFECTIVEMENT independants, "
                f"m = {self.correction_effective.n_trials}) : "
                f"{self.correction_effective.n_survivors} survivante(s). "
                "Cette borne n'est PAS le critere de retenue — elle indique seulement "
                "si le rejet tient a la severite du comptage ou a la faiblesse de la preuve."
            )
        if self.missing_peers:
            lines.append(
                "Facteur dollar INDISPONIBLE : "
                + ", ".join(self.missing_peers)
                + " absents du disque. Les familles inter-actifs n'ont donc ete testees "
                "que sur l'axe risk-on/risk-off, pas sur le dollar."
            )
        return "\n".join(lines)


def load_in_sample(symbol: str, timeframe: str) -> pd.DataFrame:
    """Trame BID restreinte a la fenetre in-sample du symbole."""
    df = snapshot.load(registry.require(symbol, timeframe, "BID"), copy=False)
    start = in_sample_start(symbol)
    return df[(df.index >= start) & (df.index < HOLDOUT_START)]


def build_market(
    symbols: Sequence[str], timeframe: str
) -> tuple[dict[str, SymbolData], dict[str, pd.DataFrame], dict[str, Any]]:
    """Donnees de marche pretes a l'execution, au cout reel."""
    market: dict[str, SymbolData] = {}
    frames: dict[str, pd.DataFrame] = {}
    models: dict[str, Any] = {}
    for symbol in symbols:
        df = load_in_sample(symbol, timeframe)
        model = costs.cost_model(symbol, timeframe)
        market[symbol] = SymbolData.from_frame(
            symbol, df, model.aligned_to(pd.DatetimeIndex(df.index))
        )
        frames[symbol] = df
        models[symbol] = model
    return market, frames, models


def run(
    symbols: Sequence[str],
    timeframe: str = "H1",
    *,
    n_control: int = N_RANDOM_CONTROL,
    n_boot: int = N_BOOTSTRAP,
    seed: int = SEED,
    alpha: float = 0.05,
    journal_path: Path = protocol.JOURNAL_PATH,
    exec_config: ExecConfig | None = None,
    progress: bool = False,
) -> ExplorationResult:
    """Execute toutes les familles sur tous les symboles et rend le verdict corrige."""
    geometry = GEOMETRY.get(timeframe, GEOMETRY["H1"])
    cfg = exec_config or ExecConfig(max_concurrent=1, max_per_symbol=1)

    market, frames, models = build_market(symbols, timeframe)
    features = {s: build_features(frames[s], geometry) for s in symbols}

    known_peers = set(registry.tradable(timeframe))
    missing = tuple(s for s in ("EURUSD", "GBPUSD", "USDJPY") if s not in known_peers)

    rows: list[dict[str, Any]] = []
    diversity_report: diversity.DiversityReport | None = None
    for symbol in symbols:
        peers = [p for p in symbols if p != symbol]
        ctx = Context(
            symbol=symbol,
            timeframe=timeframe,
            df=frames[symbol],
            features=features[symbol],
            cost=models[symbol],
            geometry=geometry,
            peers={p: frames[p] for p in peers},
        )
        single = {symbol: market[symbol]}
        families = all_families(peers)
        if diversity_report is None:
            # Mesuree une fois, sur le premier symbole : la redondance est une
            # propriete du CATALOGUE, pas de l'instrument.
            diversity_report = diversity.analyse(ctx, families)

        for family in families:
            label = f"{family_label(family)}@{symbol}"

            # ENREGISTREMENT AVANT LECTURE DU RESULTAT — ordre non negociable.
            protocol.record_trial(
                label,
                family=family_label(family),
                params={**family.parameters(), "geometrie": geometry.label},
                universe=[symbol],
                timeframe=timeframe,
                window=f"{in_sample_start(symbol).date()}..{HOLDOUT_START.date()}",
                path=journal_path,
            )

            orders: list[Order] = family.generate(ctx)
            if progress:
                print(f"  {label:44s} {len(orders):6d} candidats", flush=True)

            # Geometrie du temoin recalculee AU HASARD DE SA PROPRE BARRE.
            #
            # Sans cela, le temoin herite des distances de stop absolues du signal reel
            # — distances derivees de l'ATR au moment du signal — et les transplante a
            # une date tiree au sort. Si la volatilite y est plus forte, le meme stop
            # absolu est plus proche en multiples d'ATR : le temoin se fait sortir plus
            # souvent, pour une raison qui n'a rien a voir avec la qualite du signal.
            # Le biais joue systematiquement en faveur de la strategie et suffit a
            # produire des `bat_hasard` de 1,000 sans le moindre edge.
            def geometry_fn(
                _symbol: str, position: int, c: Context = ctx
            ) -> tuple[float, float] | None:
                return c.geometry_at(position)

            gate, result = protocol.evaluate(
                label,
                orders,
                single,
                cfg=cfg,
                n_boot=n_boot,
                n_control=n_control,
                seed=seed,
                geometry=geometry_fn,
            )
            stats = gate.details["stats"]
            control = gate.details["temoin"]
            rows.append(
                {
                    "famille": family_label(family),
                    "symbole": symbol,
                    "candidats": len(orders),
                    "trades": stats["trades"],
                    "R_moyen": stats.get("R_moyen"),
                    "IC95": stats.get("IC95"),
                    "taux_reussite": stats.get("taux_reussite"),
                    "drawdown_R": stats.get("drawdown_max_R"),
                    "taux_ambigu": stats.get("taux_ambigu"),
                    "R_cout_majore": gate.details["cout_majore"]["R_moyen"],
                    "bat_hasard": control.get("bat_hasard"),
                    "p_value": control.get("p_value"),
                    "portes_franchies": sum(gate.checks.values()),
                    "gate": gate.passed,
                    "echecs": ", ".join(gate.failed_checks),
                    "_result": result,
                }
            )

    table = pd.DataFrame(rows)
    labels = [f"{r['famille']}@{r['symbole']}" for r in rows]
    pvalues = [r["p_value"] if r["p_value"] is not None else float("nan") for r in rows]

    n_journal = protocol.n_trials(journal_path)
    correction = benjamini_hochberg(labels, pvalues, alpha=alpha, n_trials=n_journal)

    # Seconde lecture, informative : si le catalogue est redondant, le compte brut
    # surestime le nombre de chances reelles de tomber sur un faux positif. On publie
    # donc aussi la correction sur le nombre effectif — sans jamais s'en servir comme
    # critere de retenue, pour qu'aucun resultat ne puisse etre sauve en declarant
    # apres coup que ses tests etaient redondants.
    correction_effective = None
    if diversity_report is not None and diversity_report.n_families > 1:
        shrink = diversity_report.n_effective / diversity_report.n_families
        correction_effective = benjamini_hochberg(
            labels, pvalues, alpha=alpha, n_trials=max(int(round(n_journal * shrink)), len(labels))
        )
    table["q_value"] = list(correction.qvalues)
    # Survivre = franchir toutes les portes ET resister a la correction multiple.
    table["survit_correction"] = [
        bool(ok and gate) for ok, gate in zip(correction.rejected, table["gate"], strict=True)
    ]

    return ExplorationResult(
        table=table.drop(columns=["_result"]),
        correction=correction,
        timeframe=timeframe,
        symbols=tuple(symbols),
        missing_peers=missing,
        correction_effective=correction_effective,
        diversity_report=diversity_report,
    )
