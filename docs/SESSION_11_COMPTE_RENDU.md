# Session 11 — La porte de faisabilité du scalping est fermée, et on sait pourquoi

**Date :** 2026-08-11 | **Hold-out (≥ 2023-01-01) : toujours vierge.**

## La question

Le plan prévoyait une porte à franchir avant d'écrire la moindre stratégie M1/M5 : le
coût d'exécution tient-il dans l'amplitude disponible ? Le seuil était gelé depuis la
Session 4 (`SCALPING_MAX_COST_FRAC_OF_MFE = 0.25`) mais la porte n'avait jamais été
exécutée, faute de snapshots M1 et M5.

**Elle n'en avait pas besoin.** C'est le résultat de méthode de cette session.

## Pourquoi les données M1 n'étaient pas nécessaires

Ce qui coûte, ce n'est pas la taille des bougies affichées : c'est le **temps passé en
position**. Un trade de 60 minutes paie le même spread et voit la même amplitude, qu'on
le regarde sur une grille M15 ou H1.

C'est une affirmation vérifiable, et elle a été vérifiée. Sur les trois durées que nos
deux grilles couvrent en commun :

| Durée | Coût mesuré via H1 | Coût mesuré via M15 | Écart |
|---|---|---|---|
| 60 min | 0,2219 | 0,2191 | 1,3 % |
| 120 min | 0,1527 | 0,1510 | 1,1 % |
| 240 min | 0,1039 | 0,1029 | 0,9 % |

*(or ; sur le Nasdaq les écarts sont de 1,8 à 2,7 %)*

Les deux grilles disent la même chose. La grandeur ne dépend donc que de la durée, et
descendre en M1 ne révélerait rien qu'on ne puisse déjà calculer.

## La loi, mesurée puis résolue

Le spread reste constant en unités de prix quand on raccourcit la détention, tandis que
l'amplitude croît comme la racine du temps. La fraction de coût suit donc
`k / √durée`. Mesurée sur cinq durées et deux grilles, elle donne `k = 1,66` sur l'or
et `k = 1,43` sur le Nasdaq. La durée minimale viable se résout directement :
`T* = (k / 0,25)²`.

## Le verdict

| Durée visée | Or : coût | Or : réussite exigée | Nasdaq : coût | Nasdaq : réussite exigée |
|---|---|---|---|---|
| **1 min** | 166 % | **133 %** | 143 % | **121 %** |
| **5 min** | 74 % | **87 %** | 64 % | **82 %** |
| 15 min | 46 % | 73 % | 41 % | 70 % |
| 30 min | 32 % | 66 % | 28 % | 64 % |
| **60 min** | **22 %** | **61 %** | **19 %** | **60 %** |
| 240 min | 10 % | 55 % | 9 % | 54 % |

*Durée minimale viable : **44 minutes** sur l'or, **33 minutes** sur le Nasdaq.*

La colonne « réussite exigée » est la clé de lecture. Elle vient d'une identité, pas
d'un modèle : pour un trade qui vise `x`, risque `x` et paie `c`, l'équilibre impose
`p = 0,5 + c/(2x)`. C'est le taux de réussite qu'il faut atteindre pour **ne rien
gagner**. Il ne dépend d'aucune stratégie, d'aucun indicateur, d'aucun réglage.

Sur une minute, il vaut 133 % sur l'or. Un taux de réussite ne peut pas dépasser 100 %.
**Le scalping à la minute sur ce flux n'est pas difficile : il est arithmétiquement
impossible.** Sur cinq minutes il faudrait 87 % de réussite, ce qu'aucune stratégie
publique n'a jamais soutenu sur la durée.

## Ce que cela dit des robots de scalping

Cette porte explique en une ligne pourquoi les robots vendus comme « scalpeurs » ont
tous le même profil de résultat. Une courbe d'équité régulière sur un marché où le
péage est de 87 % ne peut avoir que trois origines :

1. **Un backtest sans spread réel**, ou avec un spread forfaitaire optimiste. Nous
   avons déjà rencontré ce cas dans ce dépôt : la seule stratégie qui semblait vivante
   dans l'historique (+0,06R au forfait) est passée à −0,21R au spread réel.
2. **Pas de stop, ou une grille/martingale**. Le taux de réussite affiché monte à 95 %
   parce que les pertes ne sont jamais réalisées — elles s'accumulent en positions
   ouvertes jusqu'à un événement qui solde tout.
3. **Une exécution que le particulier n'a pas** : colocation, flux direct, rabais de
   liquidité. À ce niveau le spread payé n'est plus celui-ci, et l'arithmétique change.
   Ce n'est pas le même métier.

Les deux premières sont des artefacts. La troisième est réelle mais inaccessible ici.

## Conséquence sur la suite du projet

Le mot « scalping » sort du périmètre, et c'est cohérent avec ce que l'exploration
montrait déjà sans qu'on l'ait relié : la seule configuration à franchir les cinq
portes en dix campagnes — `cassure[aucun]/suiveur1.0R` sur le Nasdaq — a une **durée de
détention médiane de 2 barres H1, soit 2 heures**. Elle vit très au-dessus du seuil de
33 minutes. Le protocole avait déjà, sans le dire, éliminé le court terme.

La cible réaliste est donc le **day trading à détention de 1 à 4 heures**, ce que le
système fait déjà. Ce n'est pas un repli : c'est le seul régime où le péage du marché
laisse de la place à un edge.

## Livrables

- `backtest/feasibility.py` — excursion favorable neutre en direction, loi de durée,
  confrontation des grilles, table de viabilité
- `alphalab scalping` — commande CLI
- `tests/unit/test_faisabilite.py` — 11 tests, dont les cas calculables à la main
- `docs/PREREGISTRE.md` § 7 — verdict inscrit sous le seuil qui l'a produit
