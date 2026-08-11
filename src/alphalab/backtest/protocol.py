"""Protocole de validation : le droit d'affirmer quelque chose.

Une esperance-R positive n'est pas une decouverte. Pour qu'une hypothese ait le droit
de figurer dans un rapport comme "interessante", elle doit franchir toutes les portes
ci-dessous. Le gate est volontairement difficile : sur ce projet, trois sessions de
recherche ont produit huit resultats negatifs, et le seul signal qui avait semble
vivant etait un artefact de cout.

Le journal des essais est l'autre moitie du dispositif. Explorer largement multiplie
les faux positifs : a 5 % de seuil, 100 configurations testees en produisent ~5 par
pur hasard. On ne peut corriger ce biais que si l'on sait combien d'essais ont ete
faits — d'ou un journal append-only ecrit AVANT de lire le resultat.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from alphalab.backtest.engine import BacktestResult, ExecConfig, Order, SymbolData, run
from alphalab.backtest.metrics import GeometryFn, beat_random, describe
from alphalab.config import (
    DOCS_DIR,
    GATE_COST_STRESS_MULT,
    GATE_MIN_BEAT_RANDOM,
    GATE_MIN_TRADES,
    N_BOOTSTRAP,
    N_RANDOM_CONTROL,
    SEED,
)

JOURNAL_PATH = DOCS_DIR / "journal_essais.jsonl"


@dataclass(frozen=True, slots=True)
class GateResult:
    """Verdict detaille d'une hypothese."""

    label: str
    checks: dict[str, bool]
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(self.checks.values())

    @property
    def failed_checks(self) -> list[str]:
        return [name for name, ok in self.checks.items() if not ok]

    def verdict(self) -> str:
        if self.passed:
            return "INTERESSANT (sous reserve de correction pour tests multiples)"
        return "REJETE : " + ", ".join(self.failed_checks)


def with_cost_multiplier(market: Mapping[str, SymbolData], mult: float) -> dict[str, SymbolData]:
    """Copie du marche dont les couts sont multiplies (test de sensibilite)."""
    out: dict[str, SymbolData] = {}
    for symbol, data in market.items():
        out[symbol] = SymbolData(
            symbol=data.symbol,
            index=data.index,
            open=data.open,
            high=data.high,
            low=data.low,
            close=data.close,
            cost=data.cost * mult,
            _pos={ts: i for i, ts in enumerate(data.index)},
        )
    return out


def evaluate(
    label: str,
    orders: Sequence[Order],
    market: Mapping[str, SymbolData],
    *,
    cfg: ExecConfig | None = None,
    n_boot: int = N_BOOTSTRAP,
    n_control: int = N_RANDOM_CONTROL,
    seed: int = SEED,
    geometry: GeometryFn | None = None,
) -> tuple[GateResult, BacktestResult]:
    """Passe une hypothese au protocole complet et renvoie son verdict.

    Portes appliquees :
      1. taille d'echantillon suffisante ;
      2. esperance-R strictement positive au cout reel ;
      3. intervalle de confiance bootstrap excluant zero ;
      4. superiorite sur le temoin aleatoire apparie (densite, biais, heure, geometrie) ;
      5. robustesse au cout : R reste positif si le spread est majore.
    """
    cfg = cfg or ExecConfig()
    result = run(orders, market, cfg)
    stats = describe(result, n_boot=n_boot, seed=seed)

    checks: dict[str, bool] = {}
    details: dict[str, Any] = {"stats": stats}

    n_trades = int(stats["trades"])
    checks["echantillon_suffisant"] = n_trades >= GATE_MIN_TRADES

    r_mean = float(stats["R_moyen"]) if n_trades else float("nan")
    checks["R_positif"] = bool(np.isfinite(r_mean) and r_mean > 0)
    checks["IC_exclut_0"] = bool(stats["exclut_0"]) and checks["R_positif"]

    if n_trades:
        control = beat_random(
            result, orders, market, n_control=n_control, seed=seed, cfg=cfg, geometry=geometry
        )
        details["temoin"] = control
        bh = control["bat_hasard"]
        checks["bat_hasard"] = bool(np.isfinite(bh) and bh >= GATE_MIN_BEAT_RANDOM)
    else:
        details["temoin"] = {"bat_hasard": float("nan"), "n_temoins": 0}
        checks["bat_hasard"] = False

    stressed_market = with_cost_multiplier(market, GATE_COST_STRESS_MULT)
    stressed = run(orders, stressed_market, cfg)
    r_stressed = float(stressed.trades["r"].mean()) if stressed.n_trades else float("nan")
    details["cout_majore"] = {
        "multiplicateur": GATE_COST_STRESS_MULT,
        "R_moyen": round(r_stressed, 4) if np.isfinite(r_stressed) else None,
        "trades": stressed.n_trades,
    }
    checks["robuste_au_cout"] = bool(np.isfinite(r_stressed) and r_stressed > 0)

    return GateResult(label=label, checks=checks, details=details), result


# --------------------------------------------------------------------------------------
# Journal des essais (append-only)
# --------------------------------------------------------------------------------------


def record_trial(
    label: str,
    *,
    family: str,
    params: Mapping[str, Any],
    universe: Sequence[str],
    timeframe: str,
    window: str,
    path: Path = JOURNAL_PATH,
) -> dict[str, Any]:
    """Enregistre une configuration testee, AVANT d'en lire le resultat.

    L'ordre compte : enregistrer apres coup permettrait d'oublier les essais rates,
    ce qui rendrait toute correction pour tests multiples cosmetique.
    """
    entry: dict[str, Any] = {
        "ts_utc": datetime.now(UTC).isoformat(),
        "label": label,
        "famille": family,
        "params": dict(params),
        "univers": list(universe),
        "timeframe": timeframe,
        "fenetre": window,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
    return entry


def read_trials(path: Path = JOURNAL_PATH) -> list[dict[str, Any]]:
    """Relit le journal des essais. Renvoie une liste vide s'il n'existe pas."""
    if not path.is_file():
        return []
    entries: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            entries.append(json.loads(line))
    return entries


def n_trials(path: Path = JOURNAL_PATH) -> int:
    """Nombre total de configurations testees — le `m` de la correction multiple."""
    return len(read_trials(path))
