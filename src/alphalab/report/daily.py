"""Plan de trading quotidien, gradé A/B/C.

Exigence de conception : **un plan est emis chaque jour, sans exception.** Les prix
bougent tous les jours, donc il y a toujours quelque chose a dire. Ce qui varie n'est pas
l'existence du plan mais la TAILLE qu'il recommande — et la probabilite calibree qui la
justifie.

Correction point-dans-le-temps : le modele qui note les candidats du jour J n'est
entraine que sur des trades CLOS avant J. Reentrainer sur l'historique complet donnerait
un plan de rejeu flatteur et inexploitable en reel.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd

from alphalab.alpha.base import GEOMETRY, Context, build_features
from alphalab.alpha.registry import all_families, family_label
from alphalab.backtest.engine import SymbolData
from alphalab.config import GRADE_RISK, HOLDOUT_START, in_sample_start
from alphalab.data import costs, registry, snapshot
from alphalab.features import sessions
from alphalab.model import calibration, dataset, walkforward
from alphalab.risk import portfolio, sizing


@dataclass(frozen=True, slots=True)
class PlanLine:
    """Une position proposee."""

    symbol: str
    family: str
    signal_ts: pd.Timestamp
    direction: int
    reference_price: float
    stop: float
    target: float
    grade: str
    risk_fraction: float
    probability: float
    expected_r: float
    rationale: str

    @property
    def side(self) -> str:
        return "ACHAT" if self.direction == 1 else "VENTE"


@dataclass(slots=True)
class DailyPlan:
    """Plan du jour : lignes retenues, lignes ecartees, et pourquoi."""

    date: pd.Timestamp
    timeframe: str
    lines: list[PlanLine] = field(default_factory=list)
    rejected: list[tuple[PlanLine, str]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    portfolio_risk: float = 0.0
    duplicates: list[tuple[str, str, float]] = field(default_factory=list)
    cost_warnings: list[str] = field(default_factory=list)

    @property
    def best_grade(self) -> str:
        return min((line.grade for line in self.lines), default="-")

    def to_console(self) -> str:
        out: list[str] = []
        out.append(f"=== PLAN DU {self.date.date()} ({self.timeframe}) ===")

        if not self.lines:
            out.append(
                "\nAucun setup declenche aujourd'hui par les 18 familles suivies.\n"
                "Forcer une position reviendrait a inventer un signal : ce serait la "
                "seule facon de degrader le resultat a coup sur."
            )
        else:
            rows = [
                {
                    "grade": line.grade,
                    "symbole": line.symbol,
                    "sens": line.side,
                    "famille": line.family,
                    "signal": str(pd.Timestamp(line.signal_ts).strftime("%H:%M")),
                    "reference": round(line.reference_price, 3),
                    "stop": round(line.stop, 3),
                    "objectif": round(line.target, 3),
                    "proba": f"{line.probability:.1%}",
                    "E[R]": f"{line.expected_r:+.3f}",
                    "risque": f"{line.risk_fraction:.2%}",
                }
                for line in self.lines
            ]
            with pd.option_context("display.width", 200):
                out.append("\n" + pd.DataFrame(rows).to_string(index=False))

            out.append("\nJustification ligne a ligne :")
            for line in self.lines:
                out.append(f"  [{line.grade}] {line.symbol} {line.side} — {line.rationale}")

        if self.rejected:
            out.append("\nEcartes par les plafonds de portefeuille :")
            for line, reason in self.rejected:
                out.append(f"  {line.symbol} {line.side} ({line.family}) — {reason}")

        if self.cost_warnings:
            out.append("\nATTENTION — fenetre d'execution structurellement chere :")
            for warning in self.cost_warnings:
                out.append(f"  {warning}")

        if self.duplicates:
            out.append("\nATTENTION — paris redondants detectes :")
            for a, b, rho in self.duplicates:
                out.append(
                    f"  {a} et {b} evoluent ensemble (correlation effective {rho:+.2f}) : "
                    "ce sont deux fois le meme pari, pas de la diversification."
                )

        out.append(f"\nRisque de portefeuille ajuste de la correlation : {self.portfolio_risk:.3%}")
        for note in self.notes:
            out.append(f"\n{note}")
        return "\n".join(out)


def _model_for_date(
    ctx: Context,
    market: dict[str, SymbolData],
    families: list[Any],
    cutoff: pd.Timestamp,
) -> tuple[calibration.CalibratedModel | None, pd.DataFrame, list[str]]:
    """Entraine le modele sur les seuls trades CLOS avant `cutoff`."""
    notes: list[str] = []
    data = dataset.build(ctx, families, market)
    if data.empty:
        return None, data, ["Aucun candidat historique : impossible de calibrer."]

    past = data[pd.DatetimeIndex(data["exit_ts"]) < cutoff].reset_index(drop=True)
    if len(past) < 300:
        notes.append(
            f"Historique insuffisant pour calibrer ({len(past)} trades clos avant "
            f"{cutoff.date()}). Les probabilites affichees retombent sur le taux de base."
        )
        return None, past, notes

    X, y = dataset.feature_matrix(past)
    splits = list(
        walkforward.purged_walk_forward(past["entry_ts"], past["exit_ts"], n_splits=5)
    )
    model = calibration.fit(X, y, splits)
    if model is None:
        notes.append("Aucun pli exploitable : probabilites ramenees au taux de base.")
    elif not model.is_useful:
        notes.append(
            f"Le modele ({model.diagnostics['modele']}) n'apporte RIEN d'exploitable : gain "
            f"de Brier de {model.skill:+.2%} sur la prediction constante "
            f"(Brier {model.diagnostics['brier']:.4f} vs {model.diagnostics['brier_base']:.4f}, "
            f"n = {model.diagnostics['n_evaluation']}). Les probabilites affichees ne valent "
            "donc pas mieux que le taux de base. Les grades qui en decoulent sont a lire "
            "comme un classement relatif, pas comme des probabilites de gain fiables."
        )
    return model, past, notes


def build(
    date: pd.Timestamp | str,
    timeframe: str = "H1",
    *,
    symbols: list[str] | None = None,
) -> DailyPlan:
    """Construit le plan d'une journee de trading donnee."""
    # Accepte aussi bien une chaine qu'un Timestamp deja localise.
    raw = pd.Timestamp(date)
    day = (raw.tz_localize("UTC") if raw.tzinfo is None else raw.tz_convert("UTC")).normalize()
    symbols = symbols or registry.tradable(timeframe)
    geometry = GEOMETRY.get(timeframe, GEOMETRY["H1"])
    plan = DailyPlan(date=day, timeframe=timeframe)

    if day >= HOLDOUT_START:
        plan.notes.append(
            f"ATTENTION : {day.date()} appartient au hold-out (>= {HOLDOUT_START.date()}). "
            "Ce plan est un rejeu de demonstration ; l'utiliser pour juger le systeme "
            "brulerait l'echantillon reserve au verdict final."
        )

    frames: dict[str, pd.DataFrame] = {}
    market: dict[str, SymbolData] = {}
    cost_models: dict[str, Any] = {}
    for symbol in symbols:
        df = snapshot.load(registry.require(symbol, timeframe, "BID"), copy=False)
        df = df[(df.index >= in_sample_start(symbol)) & (df.index < day + pd.Timedelta(days=1))]
        if df.empty:
            continue
        cost = costs.cost_model(symbol, timeframe)
        frames[symbol] = df
        cost_models[symbol] = cost
        market[symbol] = SymbolData.from_frame(
            symbol, df, cost.aligned_to(pd.DatetimeIndex(df.index))
        )

    if not frames:
        plan.notes.append("Aucune donnee disponible pour cette date.")
        return plan

    candidates: list[PlanLine] = []
    returns: dict[str, pd.Series] = {}

    for symbol, df in frames.items():
        returns[symbol] = np.log(df["close"]).diff()
        peers = [p for p in frames if p != symbol]
        ctx = Context(
            symbol=symbol,
            timeframe=timeframe,
            df=df,
            features=build_features(df, geometry),
            cost=cost_models[symbol],
            geometry=geometry,
            peers={p: frames[p] for p in peers},
        )
        families = all_families(peers)
        clf, _, notes = _model_for_date(ctx, {symbol: market[symbol]}, families, day)
        plan.notes.extend(f"[{symbol}] {n}" for n in notes)
        if clf is not None:
            plan.diagnostics[symbol] = clf.diagnostics

        trading_day = sessions.trading_day(pd.DatetimeIndex(df.index))
        today = trading_day == day

        for family in families:
            signal = family.signal(ctx)
            fired = signal[today & (signal != 0)]
            for raw_ts, direction in fired.items():
                ts = pd.Timestamp(raw_ts)  # type: ignore[arg-type]
                position = int(np.flatnonzero(df.index == ts)[0])
                geo = ctx.geometry_at(position)
                if geo is None:
                    continue
                stop_distance, target_distance = geo
                reference = float(df["close"].to_numpy()[position])
                d = int(direction)

                if clf is not None:
                    row = ctx.features.loc[[ts]].copy()
                    row["famille"] = family_label(family)
                    row["label"] = 0
                    X, _ = dataset.feature_matrix(row)
                    trained_columns = getattr(clf.pipeline, "feature_names_in_", None)
                    if trained_columns is not None:
                        X = X.reindex(columns=list(trained_columns), fill_value=0.0)
                    probability = float(clf.predict_proba(X)[0])
                else:
                    # Sans modele, la seule probabilite defendable est celle qui rend
                    # l'esperance nulle pour ce rapport gain/risque : on n'invente rien.
                    probability = 1.0 / (1.0 + geometry.tp_atr / geometry.sl_atr)

                candidates.append(
                    PlanLine(
                        symbol=symbol,
                        family=family_label(family),
                        signal_ts=ts,
                        direction=d,
                        reference_price=reference,
                        stop=reference - d * stop_distance,
                        target=reference + d * target_distance,
                        grade="C",
                        risk_fraction=0.0,
                        probability=probability,
                        expected_r=0.0,
                        rationale="",
                    )
                )

    if not candidates:
        return plan

    reward_ratio = geometry.tp_atr / geometry.sl_atr
    raw_expected = np.array(
        [sizing.expected_r(c.probability, reward_ratio) for c in candidates], dtype=float
    )
    ranks = sizing.percentiles(raw_expected)

    graded: list[PlanLine] = []
    for candidate, percentile in zip(candidates, ranks, strict=True):
        verdict = sizing.grade_candidate(
            candidate.probability, reward_ratio, percentile=float(percentile)
        )
        graded.append(
            replace(
                candidate,
                grade=verdict.grade,
                risk_fraction=verdict.risk_fraction,
                expected_r=verdict.expected_r,
                rationale=verdict.rationale,
            )
        )

    graded.sort(key=lambda line: line.expected_r, reverse=True)
    correlations = portfolio.correlation_matrix(returns)
    check = portfolio.apply_caps(
        [portfolio.Exposure(line.symbol, line.direction, line.risk_fraction) for line in graded],
        correlations,
    )

    kept = {(e.symbol, e.direction) for e in check.accepted}
    seen: set[tuple[str, int]] = set()
    for line in graded:
        key = (line.symbol, line.direction)
        if key in kept and key not in seen:
            plan.lines.append(line)
            seen.add(key)
        else:
            reason = next(
                (
                    r
                    for e, r in check.rejected
                    if e.symbol == line.symbol and e.direction == line.direction
                ),
                "non retenu apres classement",
            )
            plan.rejected.append((line, reason))

    plan.portfolio_risk = check.risk
    plan.duplicates = check.duplicates
    plan.cost_warnings = _cost_warnings(plan.lines, cost_models)

    if plan.lines and all(line.grade == "C" for line in plan.lines):
        plan.notes.append(
            "Journee faible : aucun candidat d'esperance positive. Le plan reste emis, "
            "a taille minimale — c'est la reponse honnete a la contrainte 'une position "
            "chaque jour', et elle vous montre POURQUOI le jour est faible."
        )
    return plan


def _cost_warnings(
    lines: list[PlanLine], cost_models: dict[str, Any], *, factor: float = 2.0
) -> list[str]:
    """Signale les entrees tombant dans une heure structurellement chere.

    Sur l'or, la mediane du spread triple a 22h UTC (rollover). Une entree a cette
    heure paie donc trois fois le cout normal, ce qui suffit a retourner le signe de
    l'esperance — c'est exactement ce qui avait detruit le seul signal apparemment
    vivant de l'historique de ce projet. Le plan le dit au lieu de le laisser passer.
    """
    warnings: list[str] = []
    for line in lines:
        model = cost_models.get(line.symbol)
        if model is None:
            continue
        spread = model.spread.dropna()
        if spread.empty:
            continue
        hour = pd.Timestamp(line.signal_ts).hour
        hourly = spread[pd.DatetimeIndex(spread.index).hour == hour]
        if hourly.empty:
            continue
        ratio = float(hourly.median()) / float(spread.median())
        if ratio >= factor:
            warnings.append(
                f"{line.symbol} {line.side} a {hour:02d}h UTC : spread median "
                f"{ratio:.1f}x la mediane globale. A ce cout, l'esperance affichee "
                f"({line.expected_r:+.3f}R) est probablement surestimee."
            )
    return warnings


def compare_graded_vs_forced(
    plans: list[DailyPlan],
) -> pd.DataFrame:
    """Compare la variante graduee a la variante 'taille pleine tous les jours'.

    Chiffre exactement ce que coute la contrainte "toujours une position pleine", au
    lieu d'en debattre.
    """
    rows = []
    for plan in plans:
        if not plan.lines:
            continue
        best = plan.lines[0]
        graded_r = sum(line.expected_r * line.risk_fraction for line in plan.lines)
        forced_r = best.expected_r * max(GRADE_RISK.values())
        rows.append(
            {
                "date": plan.date.date(),
                "grade": best.grade,
                "E[R]_gradee": round(graded_r, 5),
                "E[R]_forcee": round(forced_r, 5),
                "ecart": round(graded_r - forced_r, 5),
            }
        )
    return pd.DataFrame(rows)
