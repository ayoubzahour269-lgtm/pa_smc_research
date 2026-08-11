"""Configuration centrale : chemins, decoupage temporel, parametres geles.

Regle du projet : aucune constante de recherche ne doit vivre ailleurs qu'ici ou
dans un fichier de pre-enregistrement. Les scripts ne codent plus de seuils en dur.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pandas as pd

# --------------------------------------------------------------------------------------
# Chemins
# --------------------------------------------------------------------------------------

_MARKER = "data/snapshots"


def _discover_root() -> Path:
    """Racine du depot.

    Priorite a la variable d'environnement ALPHALAB_ROOT (utile quand le paquet est
    installe ailleurs que dans le depot), sinon remontee depuis ce fichier jusqu'au
    repertoire contenant `data/snapshots`.
    """
    env = os.environ.get("ALPHALAB_ROOT")
    if env:
        return Path(env).resolve()
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / _MARKER).is_dir():
            return parent
    # Repli : racine du depot dans la disposition src/ standard.
    return here.parents[2]


PROJECT_ROOT: Final[Path] = _discover_root()
DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
SNAPSHOT_DIR: Final[Path] = DATA_DIR / "snapshots"
MACRO_DIR: Final[Path] = DATA_DIR / "macro"
CALENDAR_DIR: Final[Path] = MACRO_DIR / "calendar"
SERIES_DIR: Final[Path] = MACRO_DIR / "series"
DOCS_DIR: Final[Path] = PROJECT_ROOT / "docs"
OUT_DIR: Final[Path] = PROJECT_ROOT / "out"

# --------------------------------------------------------------------------------------
# Decoupage temporel — GELE
# --------------------------------------------------------------------------------------
# Le hold-out n'est ouvert qu'UNE seule fois, a la toute fin du projet. Toute mesure
# qui le touche avant est une violation de protocole : elle brule l'echantillon et
# rend le chiffre final ininterpretable.

HOLDOUT_START: Final[pd.Timestamp] = pd.Timestamp("2023-01-01", tz="UTC")

# Debut d'echantillon par symbole. Le Nasdaq Dukascopy change de regime de seance
# avant 2019 (~15 barres/j H1, puis ~23 des 2019) : commencer avant melangerait deux
# marches differents sous un meme nom.
IN_SAMPLE_START: Final[dict[str, pd.Timestamp]] = {
    "XAUUSD": pd.Timestamp("2015-01-01", tz="UTC"),
    "NAS100": pd.Timestamp("2019-01-01", tz="UTC"),
}
DEFAULT_IN_SAMPLE_START: Final[pd.Timestamp] = pd.Timestamp("2015-01-01", tz="UTC")


def in_sample_start(symbol: str) -> pd.Timestamp:
    """Debut de la fenetre in-sample pour un symbole."""
    return IN_SAMPLE_START.get(symbol, DEFAULT_IN_SAMPLE_START)


# --------------------------------------------------------------------------------------
# Univers d'instruments
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SymbolSpec:
    """Description d'un instrument.

    `dukascopy_id` sert au telechargement (execute hors de ce conteneur, ou l'acces
    reseau aux donnees de marche est bloque). `usd_leg` indique comment l'instrument
    se comporte face au dollar : +1 si une hausse du symbole correspond a un dollar
    faible (XAUUSD, EURUSD), -1 si elle correspond a un dollar fort (USDJPY), 0 si
    l'exposition dollar n'est pas le facteur dominant (indices actions).
    """

    symbol: str
    label: str
    dukascopy_id: str
    kind: str  # "metal" | "fx" | "index"
    usd_leg: int
    point_value: float = 1.0


UNIVERSE: Final[dict[str, SymbolSpec]] = {
    "XAUUSD": SymbolSpec("XAUUSD", "Or spot", "INSTRUMENT_FX_METALS_XAU_USD", "metal", +1),
    "NAS100": SymbolSpec("NAS100", "Nasdaq 100", "INSTRUMENT_IDX_AMERICA_E_NQ_100", "index", 0),
    "EURUSD": SymbolSpec("EURUSD", "Euro / Dollar", "INSTRUMENT_FX_MAJORS_EUR_USD", "fx", +1),
    "GBPUSD": SymbolSpec("GBPUSD", "Livre / Dollar", "INSTRUMENT_FX_MAJORS_GBP_USD", "fx", +1),
    "USDJPY": SymbolSpec("USDJPY", "Dollar / Yen", "INSTRUMENT_FX_MAJORS_USD_JPY", "fx", -1),
    "US500": SymbolSpec("US500", "S&P 500", "INSTRUMENT_IDX_AMERICA_E_SP_500", "index", 0),
}

TIMEFRAMES: Final[tuple[str, ...]] = ("M1", "M5", "M15", "H1", "H4", "D1")
SIDES: Final[tuple[str, ...]] = ("BID", "ASK")

#: Duree d'une barre par timeframe. Sert au reechantillonnage et aux controles de grille.
BAR_DURATION: Final[dict[str, pd.Timedelta]] = {
    "M1": pd.Timedelta(minutes=1),
    "M5": pd.Timedelta(minutes=5),
    "M15": pd.Timedelta(minutes=15),
    "H1": pd.Timedelta(hours=1),
    "H4": pd.Timedelta(hours=4),
    "D1": pd.Timedelta(days=1),
}

# --------------------------------------------------------------------------------------
# Parametres de mesure — GELES (pre-enregistres avant tout backtest)
# --------------------------------------------------------------------------------------

#: Repli de cout quand le spread reel est indisponible sur une barre. Volontairement
#: pessimiste : il vaut mieux sous-estimer un edge que d'en inventer un.
FALLBACK_COST_FRAC_OF_ATR: Final[float] = 0.02

#: Tirages bootstrap pour l'intervalle de confiance de l'esperance-R.
N_BOOTSTRAP: Final[int] = 10_000

#: Nombre de temoins aleatoires apparies par hypothese.
N_RANDOM_CONTROL: Final[int] = 200

#: Graine de reference. Toute mesure publiee doit etre reproductible a la graine pres.
SEED: Final[int] = 12345

#: Gain relatif de Brier minimal pour qu'un modele de probabilite soit declare utile.
#: En deca, l'apport est indistinguable du bruit d'echantillonnage et la probabilite
#: affichee ne vaut pas mieux que le taux de base.
MIN_BRIER_SKILL: Final[float] = 0.01

# Gate d'acceptation (protocole de validation, section 9 du plan).
GATE_MIN_BEAT_RANDOM: Final[float] = 0.95
GATE_COST_STRESS_MULT: Final[float] = 1.5
GATE_MIN_TRADES: Final[int] = 100

# Porte de faisabilite du scalping : si le spread median depasse cette fraction de
# l'excursion favorable mediane, le timeframe est declare non tradable.
SCALPING_MAX_COST_FRAC_OF_MFE: Final[float] = 0.25

# --------------------------------------------------------------------------------------
# Paliers de conviction et risque — PRE-ENREGISTRES
# --------------------------------------------------------------------------------------
# Un plan est emis CHAQUE jour, sans exception. Les jours sans candidat de qualite
# produisent une ligne de grade C, a taille minimale, avec sa probabilite calibree
# affichee : on voit POURQUOI le jour est faible au lieu de le subir.

#: Esperance-R minimale pour chaque palier, et fraction du capital risquee.
GRADE_A_MIN_R: Final[float] = 0.15
GRADE_B_MIN_R: Final[float] = 0.0
GRADE_RISK: Final[dict[str, float]] = {"A": 0.010, "B": 0.005, "C": 0.001}

#: Un grade A exige aussi d'etre dans le haut du classement du jour.
GRADE_A_MIN_PERCENTILE: Final[float] = 0.90

# Plafonds de portefeuille.
MAX_DAILY_LOSS_R: Final[float] = 2.0
MAX_CONCURRENT_POSITIONS: Final[int] = 3
MAX_PER_SYMBOL: Final[int] = 1
#: Plafond du risque de portefeuille ajuste de la correlation, en fraction du capital.
MAX_PORTFOLIO_RISK: Final[float] = 0.02
#: Au-dela, deux candidats sont consideres comme le meme pari.
DUPLICATE_CORRELATION: Final[float] = 0.6
