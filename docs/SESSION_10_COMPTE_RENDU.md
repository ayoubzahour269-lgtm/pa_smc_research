# Session 10 — Compte-rendu : la sortie compte, et sept anomalies documentées

**Date :** 2026-08-11 | **Hold-out (≥ 2023-01-01) : toujours vierge.**

## Deux angles morts traités

### 1. Une seule règle de sortie sur 48 configurations

Les campagnes S6 à S9 ont évalué 48 configurations avec **une seule et même sortie** :
stop fixe à 1×ATR, objectif fixe à 2×ATR, échéance 24 barres. C'est une monoculture, et
elle est dangereuse dans un sens précis : si la sortie est mal choisie, **toutes** les
entrées paraissent mauvaises, et on conclut à tort qu'aucune n'a de valeur.

Le moteur accepte désormais trois règles supplémentaires : stop suiveur, mise à seuil
de rentabilité, clôture à heure fixe.

**Piège traité explicitement.** Le stop suiveur se met à jour **en fin de barre**, pour
servir à la barre *suivante*. Le mettre à jour avant le test des barrières permettrait
à une barre d'être stoppée par un niveau dérivé de son propre plus-haut — un stop
déclenché par un prix qu'il n'avait pas encore vu. Un test golden fige ce comportement
(`test_le_suiveur_ne_se_declenche_pas_sur_sa_propre_barre`).

### 2. Sept anomalies documentées, jamais testées ici

Toutes les familles précédentes venaient de l'analyse technique praticienne. Celles-ci
viennent de travaux publiés et répliqués. **Leur intérêt méthodologique est décisif :
leurs paramètres viennent de l'extérieur.** Ils n'ont pas été choisis en regardant nos
données, ce qui élimine une source de sur-ajustement que les familles maison ne peuvent
jamais totalement écarter.

| Famille | Inspiration |
|---|---|
| `momentum_intraseance` | Gao, Han, Li, Zhou (2018) — le début de séance prédit la fin |
| `drift_overnight` (2 sens) | Lou, Polk, Skouras (2019) — le rendement s'accumule hors séance |
| `momentum_series_temporelles` | Moskowitz, Ooi, Pedersen (2012) |
| `niveaux_ronds` (rejet / cassure) | Osler (2003) — agrégation des ordres autour des chiffres ronds |
| `cassure_ratee` | cassure qui ne tient pas, distincte du balayage intrabarre |

## Le résultat le plus intéressant : la sortie change le verdict

Même entrée, quatre sorties, sur le Nasdaq :

| Sortie | Trades | R moyen | IC 95 % | Gate |
|---|---|---|---|---|
| fixe (référence S6-S9) | 778 | +0,0966 | (−0,004 ; +0,197) | non |
| **suiveur 1R** | 814 | **+0,1047** | **(+0,029 ; +0,180)** | **oui** |
| seuil de rentabilité 1R | 790 | +0,0677 | (−0,023 ; +0,156) | non |
| clôture 21h | 784 | +0,0711 | (−0,026 ; +0,169) | non |

Le stop suiveur fait passer la même entrée de « échoue » à « franchit les cinq portes ».
Mais le mécanisme mérite d'être compris exactement : **il gagne surtout en réduisant la
dispersion**, pas en augmentant l'espérance. L'espérance passe de +0,097 à +0,105 —
marginal. L'intervalle de confiance, lui, se resserre au point d'exclure zéro. C'est
cela qui ouvre le gate.

**Contre-épreuve sur l'or** : les quatre sorties donnent −0,079 à −0,089. La sortie ne
crée pas d'edge là où il n'y en a pas. Elle révèle celui qui existe, ou n'en révèle
aucun.

## Les anomalies documentées sur nos données

Toutes négatives, sauf une non significative :

| Famille | R Nasdaq | R or |
|---|---|---|
| `momentum_series_temporelles` | **+0,048** (p = 0,31) | −0,147 |
| `drift_overnight[achat]` | −0,004 | −0,148 |
| `niveaux_ronds[rejet]` | −0,021 | −0,114 |
| `niveaux_ronds[cassure]` | −0,046 | −0,085 |
| `cassure_ratee` | −0,100 | −0,052 |
| `momentum_intraseance` | −0,070 | −0,242 |
| `drift_overnight[vente]` | −0,188 | −0,364 |

Ce résultat n'invalide pas ces travaux. Il dit que ces effets, mesurés sur d'autres
marchés, d'autres périodes et d'autres horizons, **ne se transportent pas** sur un CFD
Nasdaq ou or en horaire, sur 2019-2022, après spread réel. C'est une information utile
en soi : elle vaut mieux que de supposer qu'ils s'y transportent.

### Un cas qui illustre pourquoi le gate a cinq portes

`cassure_ratee` sur l'or : p-value de 0,045 — elle bat 95,5 % des témoins — mais
espérance de **−0,052**. Elle est donc « moins mauvaise que le hasard », pas rentable.
Sans la porte « R > 0 » évaluée séparément, une p-value flatteuse aurait pu la faire
passer pour une découverte.

## Verdict global

**68 configurations évaluées, aucune retenue.**

Deux franchissent les cinq portes prises isolément — `cassure[aucun]/suiveur1.0R`
(+0,105) et `opening_range` (+0,093), toutes deux sur le Nasdaq. Aucune ne survit à la
correction sur les 74 essais journalisés.

Diversité : **34 familles déclarées, 27,0 tests effectivement indépendants**
(redondance 21 %). Deux groupes redondants identifiés, dont un nouveau :
`cassure_ratee` et `smc_balayage` — les deux parient sur un retournement après un
extrême, ce qui est cohérent.

La borne basse (correction sur les tests effectivement indépendants, m = 68) donne
également **0 survivante**. Le rejet ne tient donc pas à la sévérité du comptage.

## Ce qu'il faut retenir

1. **La sortie compte, et elle agit sur la dispersion plus que sur la moyenne.** C'est
   le mécanisme réel, souvent mal décrit. Le savoir change la façon de concevoir un
   système.
2. **La sortie ne sauve pas une mauvaise entrée** — la contre-épreuve sur l'or est
   nette.
3. **Sept effets documentés ne se transportent pas ici.** Y compris ceux que la
   littérature considère comme les plus robustes.
4. Le Nasdaq reste systématiquement moins hostile que l'or, pour la troisième campagne
   consécutive et par trois méthodes différentes.

## Ce qui reste ouvert

- **Facteur dollar** : EUR/USD, GBP/USD, USD/JPY toujours absents.
- **Macro / news** : non construit.
- **Hold-out** : fermé.

## Livrables

- `ExitPolicy` dans `backtest/engine.py` + 7 tests golden
- `alpha/families/documented.py` — 5 classes, 7 configurations
- `docs/SESSION_10_EXPLORATION_H1.md` + `.csv`
- `docs/journal_essais.jsonl` : 74 essais
