> **Note de méthode** — cette session est née d'une remarque de l'utilisateur :
> *« je crois que c'était plus des changements de paramètre que de vraies stratégies
> distinctes »*. Elle était fondée. Le premier réflexe aurait été d'argumenter ;
> le bon réflexe était de mesurer.

# Session 9 — Compte-rendu : diversité mesurée, redondance corrigée, mécanismes ajoutés

**Date :** 2026-08-11 | **Hold-out (≥ 2023-01-01) : toujours vierge.**

## Le problème

La Session 6 annonçait « 18 familles testées ». La mesure montre que trois d'entre
elles — cassure filtrée par l'ADX, cassure filtrée par le spread, et cassure nue —
partageaient le **même déclencheur** (une clôture dépassant l'extrême des N barres
précédentes) et, quand elles se déclenchaient ensemble, s'accordaient sur la direction
dans **100 % des cas**.

Trois entrées au catalogue pour une seule idée. Cela produit deux dommages distincts :

1. l'exploration paraît plus large qu'elle ne l'est ;
2. la correction pour tests multiples se durcit sans raison — des tests redondants ne
   sont pas autant de chances indépendantes de tomber sur un faux positif.

## Ce qui a été fait

### 1. Un module de mesure de diversité (`alpha/diversity.py`)

Il calcule, pour tout catalogue :

- le **chevauchement** des signaux (Jaccard) entre chaque paire de familles ;
- l'**accord de direction** sur les barres communes ;
- le regroupement des familles redondantes ;
- le **nombre effectif de tests indépendants**, par décomposition en valeurs propres
  (méthode de Li et Ji).

La distinction qui compte : deux familles peuvent se déclencher aux mêmes instants en
pariant **l'inverse**. C'est de la diversité maximale, pas de la redondance. Le module
exige donc chevauchement élevé **et** accord de direction pour regrouper. Un test dédié
vérifie ce comportement — sans lui, on classerait comme redondantes deux familles qui
parient exactement le contraire l'une de l'autre.

Vérification sur le catalogue réel : `lead_lag+` et `lead_lag−` se chevauchent à
**100 %** avec un accord de **0,00**. Le module refuse de les regrouper. C'est correct.

### 2. Fusion des familles redondantes

`momentum_conditionne`, `microstructure_liquidite` et la cassure nue deviennent une
seule famille `Breakout`, dont le filtre (`aucun` / `adx` / `compression` /
`spread_bas`) est un **paramètre déclaré**. Chaque variante reste un essai journalisé —
rien n'est caché — mais le catalogue ne prétend plus à quatre idées là où il n'y en a
qu'une.

`opening_range` et `smc_bos` restent distinctes : elles cassent aussi un extrême, mais
un extrême défini autrement (range de séance / pivot de structure confirmé), et leur
chevauchement mesuré reste sous 13 %.

### 3. Cinq mécanismes réellement absents, ajoutés

| Famille | Ce qu'elle teste, et pourquoi c'est différent |
|---|---|
| `regime_volatilite` | Le régime comme **état qui change le sens du pari** (suivre en vol. haute, fader en vol. basse), non comme filtre qui autorise ou interdit |
| `momentum_multijour` | Un **horizon de plusieurs jours**, alors que tout le reste du catalogue est intraday |
| `saisonnalite_intraday` | Moyenne historique par cellule heure × jour de semaine, estimée en **expansion causale** |
| `cointegration` | Ratio de couverture estimé sur fenêtre glissante + **demi-vie du retour à la moyenne** — deux actifs peuvent être corrélés sans jamais converger |
| `ensemble` | Vote de mécanismes **distincts** — faire voter des variantes du même indicateur ne teste rien |

Toutes passent le harnais d'anti-anticipation, sur données synthétiques et réelles.

## Résultats

**48 configurations évaluées (24 familles × 2 instruments). Aucune retenue.**

Diversité du nouveau catalogue : **24 familles déclarées, 21,0 tests effectivement
indépendants — redondance 13 %** (contre un groupe de 5 familles interchangeables
auparavant).

Les cinq nouveaux mécanismes sont tous négatifs :

| Famille | R or | R Nasdaq |
|---|---|---|
| `regime_volatilite` | −0,105 | −0,067 |
| `momentum_multijour` | −0,130 | −0,139 |
| `saisonnalite_intraday` | −0,230 | −0,065 |
| `ensemble` | −0,169 | −0,022 |
| `cointegration` | −0,445 (26 trades) | −0,444 (23 trades) |

Trois observations qui méritent d'être retenues :

**L'idée de « confluence » échoue.** La croyance est répandue : plus de signaux
d'accord, meilleure la position. Mesurée sur deux instruments, avec des mécanismes
réellement distincts pour que le vote ait un sens, elle donne −0,17 et −0,02.

**`saisonnalite_intraday` sur l'or : p-value de 1,000.** Le témoin apparié à l'heure
la bat dans 100 % des cas. C'est exactement le comportement attendu — une stratégie
fondée uniquement sur l'heure, confrontée à un témoin qui tire au hasard à la même
heure, ne peut avoir aucune information par construction. C'est une validation du
protocole autant qu'un rejet de la famille.

**`cointegration` n'est pas rejetée : elle n'est pas testée.** 23 et 26 trades, très
en dessous du minimum de 100. Le filtre de demi-vie est très restrictif, et surtout
l'or et le Nasdaq n'ont aucune raison économique d'être cointégrés. Cette famille
attend une vraie paire (or / EUR/USD, ou deux indices actions).

## Le rejet tient-il à la sévérité du comptage ?

Non, et c'est désormais publié. Le rapport donne deux corrections :

- **critère de retenue** : correction sur les 54 essais journalisés → 0 survivante ;
- **borne basse informative** : correction sur les tests effectivement indépendants
  (m = 48) → **0 survivante également**.

La borne basse n'est jamais utilisée comme critère. La raison est simple : autoriser un
résultat à être sauvé en déclarant après coup que ses tests étaient redondants ouvrirait
exactement la porte que tout ce dispositif sert à fermer.

## Ce que cette session change

- L'affirmation « on a exploré large » est désormais **chiffrée** (21 tests
  indépendants) au lieu d'être supposée.
- Toute nouvelle famille sera confrontée aux existantes avant d'être ajoutée.
- Cinq angles morts réels ont été comblés, et leur réponse est négative.

## Ce qui reste ouvert

- **Facteur dollar** : EUR/USD, GBP/USD, USD/JPY toujours absents. La cointégration et
  les familles inter-actifs n'ont pu être testées que sur l'axe risk-on/risk-off.
- **Macro / news** : non construit.
- **Ordre et volume** : le volume Dukascopy est indicatif, inexploitable en l'état.
- **Hold-out** : fermé.

## Livrables

- `src/alphalab/alpha/diversity.py` + 12 tests
- `src/alphalab/alpha/families/breakout.py` (fusion), `regime.py`, `ensemble.py`,
  `Cointegration` dans `crossasset.py`
- `docs/SESSION_9_EXPLORATION_H1.md` + `.csv`
- `docs/journal_essais.jsonl` : 54 essais
