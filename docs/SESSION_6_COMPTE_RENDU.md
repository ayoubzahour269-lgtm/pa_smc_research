# Session 6 — Compte-rendu : exploration large, 18 familles, 2 instruments

**Date :** 2026-08-11 | **Portée :** recherche offline. Aucune décision de capital.
**Hold-out (≥ 2023-01-01) : toujours vierge.**

## Question de la session

Les sessions 1 à 3 avaient testé 8 hypothèses de price action mécanique et les avaient
toutes rejetées. La question ouverte n'était pas « SMC marche-t-il ? » mais « **existe-t-il
une famille d'approches, quelle qu'elle soit, qui survive à un protocole honnête sur ces
deux instruments ?** »

D'où le changement de méthode : au lieu de coder chaque hypothèse à la main dans un script
jetable — ce qui avait borné l'exploration à 8 essais — on construit une plateforme neutre
où une piste coûte une classe, puis on explore large.

## Ce qui a été fait

18 familles × 2 instruments = **36 configurations**, toutes évaluées au spread réel, en
in-sample (or 2015-2022, Nasdaq 2019-2022), sous un protocole identique. Aucune famille n'a
eu de statut privilégié — les 4 familles SMC ont passé exactement les mêmes portes que les
14 autres.

| Thème | Familles |
|---|---|
| Temps | dérive de séance, opening range, fin de mois |
| Tendance | momentum conditionné (ADX + séance), compression→expansion, repli de tendance |
| Retour à la moyenne | écart au VWAP, gap d'ouverture, fade du range asiatique |
| Structure (SMC) | CHoCH + retracement, BOS, balayage de liquidité, retest de FVG |
| Microstructure | régime de liquidité, choc de spread |
| Inter-actifs | lead-lag (2 sens), divergence de corrélation |

## Résultat

**Aucune famille retenue sur 36.**

Une seule franchit les cinq portes du protocole prises isolément — `opening_range` sur le
Nasdaq : R = +0,093, IC95 = (+0,0014 ; +0,185), taux de réussite 38,3 %, R encore positif
(+0,075) à 1,5× le spread, témoin apparié battu dans 99 % des cas. Mais sa p-value de
0,0149 donne une **q-value de 0,32** une fois corrigée des 36 essais journalisés : très
au-dessus du seuil de 0,05. Autrement dit, trouver un résultat de cette qualité parmi 36
essais est banal.

Les deux meilleures espérances brutes — `compression_expansion` (+0,145) et `smc_bos`
(+0,129) sur le Nasdaq — échouent avant même la correction : leur intervalle de confiance
contient zéro.

## Le biais que le protocole a lui-même révélé

Le premier passage donnait `bat_hasard = 1,000` pour les quatre meilleures familles Nasdaq
— la stratégie battait les 200 témoins, sans exception. Un tel score, conjugué à des
intervalles de confiance contenant zéro, était incohérent et a déclenché une vérification.

**Cause trouvée :** le témoin aléatoire héritait des distances de stop **absolues** du
signal réel, dérivées de l'ATR au moment du signal, puis les transplantait à une date tirée
au sort. Quand la volatilité y était plus forte, ce même stop absolu se trouvait plus près
en multiples d'ATR : le témoin se faisait sortir plus souvent, pour une raison sans aucun
rapport avec la qualité du signal. Le biais jouait systématiquement en faveur de la
stratégie.

Après correction — le témoin recalcule sa géométrie de risque sur sa propre barre — les
`bat_hasard` retombent à 0,975-0,99 et **le verdict passe de « 1 famille retenue » à
« aucune »**. Le biais était donc matériel, pas cosmétique.

C'est le résultat le plus utile de la session : le protocole a détecté une erreur dans le
protocole lui-même.

## Un second piège, corrigé au passage

La correction de Benjamini-Hochberg ne juge que la p-value du témoin — **pas le signe de
l'espérance**. Le premier rapport annonçait donc « 4 hypothèses survivantes » en listant
des familles dont certaines échouaient au gate. Le rapport distingue désormais deux
nombres, et n'annonce en premier que leur conjonction : franchir les portes **et** résister
à la correction.

## Ce que confirme cette session

- **Le Nasdaq reste moins hostile que l'or.** Les cinq meilleures espérances sur six sont
  des configurations Nasdaq. C'est cohérent avec le constat de la Session 3, obtenu par une
  méthode entièrement différente.
- **SMC n'est ni meilleur ni pire que le reste.** `smc_bos` est la deuxième meilleure
  espérance brute ; `smc_choch` et `smc_fvg` sont parmi les plus mauvaises. Rien ne
  distingue cette famille des autres, ce qui répond à la question laissée ouverte par la
  Session 3.
- **La correction pour tests multiples n'est pas une formalité.** Sans elle, cette session
  aurait publié une découverte. Avec elle, il n'y en a pas.

## Réserves

- In-sample uniquement. Ces chiffres ne disent rien de la performance future.
- Taux de barres ambiguës (stop et objectif atteignables dans la même barre) : 2,5 % au
  maximum. Faible, donc les résultats dépendent peu de l'hypothèse intrabarre — ce ne sera
  pas le cas en M1/M5.
- **Facteur dollar indisponible.** EUR/USD, GBP/USD et USD/JPY ne sont pas dans le dépôt.
  Les familles inter-actifs n'ont donc testé qu'un axe risk-on/risk-off (or ↔ Nasdaq), pas
  la relation dollar ↔ or. Cette piste reste entière, et c'est la dépendance la plus
  importante côté données.
- Une seule géométrie de risque testée (SL 1,0×ATR, TP 2,0×ATR, 24 barres). Toute variante
  compterait comme des essais supplémentaires et durcirait la correction.

## Livrables

- `src/alphalab/alpha/` — interface `AlphaFamily`, registre, 18 familles en 6 fichiers
  thématiques
- `src/alphalab/model/multipletesting.py` — p-value de permutation, Benjamini-Hochberg,
  Bonferroni
- `src/alphalab/explore.py` — runner d'exploration
- `src/alphalab/report/research.py` — rendu console et Markdown
- `docs/journal_essais.jsonl` — 36 essais, append-only, idempotent par configuration
- `docs/SESSION_6_EXPLORATION_H1.md` + `.csv` — table complète

## Prochaine étape

Deux voies, dans cet ordre :

1. **Fournir EUR/USD, GBP/USD, USD/JPY** (`alphalab freeze ...` sur votre machine). C'est ce
   qui débloque le facteur dollar et la seule piste que vous aviez identifiée par
   observation directe. Sans ces données, elle reste non testée — pas rejetée.
2. **Couche de calibration** (Session 7) : triple-barrière, walk-forward purgé, probabilité
   calibrée. Elle a une valeur indépendante du résultat ci-dessus, puisqu'elle est ce qui
   permet au plan quotidien d'afficher une probabilité honnête plutôt qu'un score.

Le hold-out reste fermé. Il ne sera ouvert qu'une fois, à la toute fin.
