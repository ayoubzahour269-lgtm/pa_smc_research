"""Probabilite calibree : transformer un score en probabilite verifiable.

Un modele qui sort 0,80 ne dit rien tant qu'on n'a pas verifie que, parmi les candidats
notes 0,80, environ 80 % gagnent effectivement. C'est la difference entre un SCORE et une
PROBABILITE, et c'est toute la difference entre un plan de trading utilisable et un
affichage rassurant.

Trois garde-fous, dans cet ordre :

  1. les predictions sont produites **hors echantillon**, par walk-forward purge ;
  2. elles sont **recalibrees** (isotonique), car les modeles sont typiquement
     sur-confiants aux extremes ;
  3. la qualite est **publiee** : score de Brier, erreur de calibration attendue (ECE) et
     diagramme de fiabilite. Un modele qui ne bat pas la prediction constante — "toujours
     le taux de base" — est declare inutile, ce qui est une information en soi.

Choix du modele : la regression logistique regularisee est la reference, le gradient
boosting le challenger, et c'est le Brier hors-pli qui tranche. Avec quelques milliers
d'observations et des features bruitees, le modele simple gagne le plus souvent — mais on
le verifie plutot que de le supposer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import pairwise
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from alphalab.config import MIN_BRIER_SKILL, SEED
from alphalab.model.walkforward import Split, coverage
from alphalab.types import BoolArray, FloatArray, IntArray


def make_models(seed: int = SEED) -> dict[str, Any]:
    """Modeles candidats. L'imputation et la mise a l'echelle sont DANS le pipeline.

    C'est essentiel : ajustees sur le pli d'apprentissage uniquement, elles ne peuvent
    pas faire fuiter de statistique globale vers le pli de test.
    """
    return {
        "logistique": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(C=0.1, max_iter=2000, random_state=seed)),
            ]
        ),
        "gradient_boosting": Pipeline(
            [
                (
                    "clf",
                    HistGradientBoostingClassifier(
                        max_depth=3,
                        max_iter=150,
                        learning_rate=0.05,
                        l2_regularization=1.0,
                        random_state=seed,
                    ),
                )
            ]
        ),
    }


def brier(y: IntArray, p: FloatArray) -> float:
    """Score de Brier — erreur quadratique moyenne sur les probabilites. Plus bas = mieux."""
    return float(np.mean((p - y) ** 2))


def expected_calibration_error(y: IntArray, p: FloatArray, *, n_bins: int = 10) -> float:
    """Ecart moyen, pondere, entre probabilite annoncee et frequence observee."""
    if y.size == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for lo, hi in pairwise(edges):
        mask = (p >= lo) & (p < hi if hi < 1.0 else p <= hi)
        if not mask.any():
            continue
        total += mask.mean() * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return float(total)


def reliability_table(y: IntArray, p: FloatArray, *, n_bins: int = 10) -> pd.DataFrame:
    """Diagramme de fiabilite sous forme de table : annonce vs observe, par tranche."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for lo, hi in pairwise(edges):
        mask = (p >= lo) & (p < hi if hi < 1.0 else p <= hi)
        if not mask.any():
            continue
        rows.append(
            {
                "tranche": f"[{lo:.1f}, {hi:.1f})",
                "n": int(mask.sum()),
                "annonce": round(float(p[mask].mean()), 4),
                "observe": round(float(y[mask].mean()), 4),
                "ecart": round(float(p[mask].mean() - y[mask].mean()), 4),
            }
        )
    return pd.DataFrame(rows)


@dataclass(slots=True)
class CalibratedModel:
    """Modele retenu, sa recalibration, et les diagnostics qui le jugent."""

    name: str
    pipeline: Any
    calibrator: IsotonicRegression | None
    base_rate: float
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def predict_proba(self, X: pd.DataFrame) -> FloatArray:
        raw = self.pipeline.predict_proba(X)[:, 1]
        if self.calibrator is None:
            return np.asarray(raw, dtype=np.float64)
        return np.asarray(self.calibrator.predict(raw), dtype=np.float64)

    @property
    def skill(self) -> float:
        """Gain relatif de Brier sur la prediction constante. 0 = aucun apport.

        Le gain ABSOLU de Brier est trompeur : passer de 0,2271 a 0,2269 est
        techniquement "meilleur" et pratiquement nul. Le rapporter en relatif rend la
        nullite visible.
        """
        base = float(self.diagnostics.get("brier_base", 0.0))
        if base <= 0:
            return 0.0
        return 1.0 - float(self.diagnostics.get("brier", np.inf)) / base

    @property
    def is_useful(self) -> bool:
        """Vrai si le modele apporte un gain NON NEGLIGEABLE hors echantillon.

        Exiger simplement "mieux que la constante" laisserait passer des gains de
        l'ordre de 0,1 %, indistinguables du bruit d'echantillonnage. Un modele qui
        n'apporte rien doit etre declare tel : c'est plus utile que de l'habiller.
        """
        return self.skill >= MIN_BRIER_SKILL


def out_of_fold_predictions(
    model: Any, X: pd.DataFrame, y: IntArray, splits: list[Split]
) -> tuple[FloatArray, BoolArray]:
    """Predictions hors pli : chaque observation est predite par un modele qui ne l'a pas vue."""
    p = np.full(len(X), np.nan)
    for split in splits:
        fitted = model
        fitted.fit(X.iloc[split.train], y[split.train])
        p[split.test] = fitted.predict_proba(X.iloc[split.test])[:, 1]
    mask = ~np.isnan(p)
    return p[mask].astype(np.float64), mask


def fit(
    X: pd.DataFrame,
    y: IntArray,
    splits: list[Split],
    *,
    seed: int = SEED,
) -> CalibratedModel | None:
    """Selectionne le meilleur modele au Brier hors-pli, puis le recalibre.

    Rend `None` si aucun pli exploitable n'a pu etre construit — cas frequent quand les
    candidats sont peu nombreux, et qu'il vaut mieux signaler que bricoler.
    """
    if not splits or len(X) == 0:
        return None

    base_rate = float(np.mean(y))
    results: dict[str, dict[str, Any]] = {}

    for name, model in make_models(seed).items():
        p_oof, mask = out_of_fold_predictions(model, X, y, splits)
        if p_oof.size == 0:
            continue
        y_oof = y[mask]
        results[name] = {
            "p": p_oof,
            "y": y_oof,
            "mask": mask,
            "brier": brier(y_oof, p_oof),
            "brier_base": brier(y_oof, np.full_like(p_oof, base_rate)),
        }

    if not results:
        return None

    best_name = min(results, key=lambda k: float(results[k]["brier"]))
    best = results[best_name]

    # Recalibration isotonique — ajustee sur la PREMIERE moitie des predictions
    # hors-pli, evaluee sur la seconde.
    #
    # Ajuster l'isotonique sur l'ensemble puis mesurer l'ECE sur ce meme ensemble donne
    # mecaniquement une erreur de calibration proche de zero : la courbe passe par les
    # points qui ont servi a la tracer. Le chiffre serait alors rassurant et faux. Le
    # decoupage temporel (et non aleatoire) preserve en outre l'ordre chronologique.
    p_oof, y_oof = best["p"], best["y"]
    cut = len(p_oof) // 2
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)

    if cut >= 50 and len(p_oof) - cut >= 50:
        calibrator.fit(p_oof[:cut], y_oof[:cut])
        p_eval = np.asarray(calibrator.predict(p_oof[cut:]), dtype=np.float64)
        y_eval = y_oof[cut:]
        calibration_note = "isotonique ajustee sur la 1re moitie, evaluee sur la 2nde"
    else:
        # Trop peu de points pour une evaluation honnete : on ne recalibre pas plutot
        # que de publier un chiffre auto-valide.
        calibrator.fit(p_oof, y_oof)
        p_eval, y_eval = p_oof, y_oof
        calibration_note = "ATTENTION : ECE auto-evalue (echantillon trop court pour scinder)"

    pipeline = make_models(seed)[best_name]
    pipeline.fit(X, y)

    brier_value = float(brier(y_eval, p_eval))
    brier_base_value = float(brier(y_eval, np.full_like(p_eval, base_rate)))
    diagnostics: dict[str, Any] = {
        "modele": best_name,
        "n_observations": int(len(X)),
        "n_hors_pli": int(p_oof.size),
        "n_evaluation": int(p_eval.size),
        "taux_de_base": round(base_rate, 4),
        "brier": round(brier_value, 5),
        "brier_base": round(brier_base_value, 5),
        "skill": round(1.0 - brier_value / brier_base_value if brier_base_value > 0 else 0.0, 5),
        "ece": round(expected_calibration_error(y_eval, p_eval), 4),
        "note_calibration": calibration_note,
        "comparaison": {k: round(float(v["brier"]), 5) for k, v in results.items()},
        **coverage(splits, len(X)),
    }
    return CalibratedModel(
        name=best_name,
        pipeline=pipeline,
        calibrator=calibrator,
        base_rate=base_rate,
        diagnostics=diagnostics,
    )


def expected_r(probability: FloatArray | float, reward_ratio: float, cost_r: float = 0.0) -> Any:
    """Esperance en R d'un candidat, depuis sa probabilite calibree.

    `E[R] = p x R_objectif - (1 - p) x 1 - cout`

    C'est ce nombre — et non un score de confluence — qui pilote le grade et la taille
    dans le plan quotidien.
    """
    p = np.asarray(probability, dtype=float)
    return p * reward_ratio - (1.0 - p) - cost_r
