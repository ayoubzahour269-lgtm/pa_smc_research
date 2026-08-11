# Architecture

## Flux de données

```
data/snapshots/*.csv + manifest        données de marché immuables (sha256)
data/macro/                            calendrier économique + séries FRED millésimées
        │
        ▼
alphalab.data      schema · snapshot · registry · costs · freeze
        │          contrat OHLCV validé, spread réel ASK.open − BID.open
        ▼
alphalab.features  indicators · resample · sessions · microstructure · seasonality
        │          smc · crossasset          → colonnes de features, sans anticipation
        ▼
alphalab.alpha     familles d'hypothèses     → Order (symbole, sens, SL, TP, échéance)
        │
        ├──────────────► alphalab.backtest   moteur portefeuille + métriques + protocole
        │                                     verdict : franchit le gate, ou non
        ▼
alphalab.model     labeling · walkforward · calibration · multipletesting
        │          → P(ce candidat gagne), calibrée hors échantillon
        ▼
alphalab.risk      sizing (paliers A/B/C) · portfolio (plafonds corrélés)
        ▼
alphalab.report    plan quotidien gradé · rapports de recherche
```

## Séparation des responsabilités

Le découpage n'est pas décoratif : chaque frontière empêche une erreur précise.

| Couche | Ce qu'elle sait | Ce qu'elle ne sait PAS |
|---|---|---|
| `data` | où sont les données, si elles sont intactes, ce que coûte une exécution | ce qu'est une stratégie |
| `features` | comment transformer des prix en colonnes | quand acheter |
| `alpha` | quand un setup est présent | s'il est rentable, quelle taille prendre |
| `backtest` | comment exécuter des ordres et les juger | d'où viennent les ordres |
| `model` | quelle est la probabilité de gain d'un candidat | comment le candidat a été trouvé |
| `risk` | combien engager | si le signal est bon |
| `report` | comment présenter | comment décider |

Conséquence utile : le moteur de backtest peut comparer sur une même échelle des
approches aussi hétérogènes qu'un effet de séance et une structure SMC, parce qu'il ne
voit d'elles que des `Order`.

## Conventions non négociables

**Immuabilité des données.** `snapshot.write` refuse d'écraser. Toute nouvelle donnée
est une nouvelle version (`v2`, `v3`…). Un résultat publié doit rester reproductible.

**Pas d'anticipation.** Toute feature doit être calculable avec les seules données
`≤ t`. Les quatre pièges du domaine :

- les pivots fractals ne sont *confirmés* que `W` barres après leur formation ;
- les zones « validées après coup » (order blocks, FVG mitigés) doivent porter leur
  date de confirmation, pas leur date de formation ;
- les séries macro sont **révisées** : utiliser la valeur finale dans un backtest de
  2019 est une anticipation classique, d'où les millésimes `realtime_start` ;
- un calendrier économique donne l'horaire à l'avance (utilisable) mais la valeur
  seulement après publication (inutilisable avant `publie_a_utc`).

La suite `tests/antilookahead/` recalcule les features sur historique tronqué et exige
l'égalité stricte. Une feature qui échoue est un bug bloquant, pas un compromis.

**Exécution conservatrice.** Entrée à l'ouverture de la barre *suivante*. Coût prélevé
au spread de la barre d'entrée. Dans une barre où stop et objectif sont tous deux
atteignables, on retient le **stop** — et le taux de ces barres ambiguës est publié,
parce qu'il mesure honnêtement l'incertitude du backtest.

**Fenêtres temporelles.** In-sample : depuis `IN_SAMPLE_START[symbole]` jusqu'à
`HOLDOUT_START`. Le Nasdaq démarre en 2019 parce que le flux Dukascopy change de régime
de séance avant (≈15 barres/jour H1 puis ≈23) — mélanger les deux reviendrait à étudier
deux marchés différents sous un même nom. Le hold-out (≥ 2023-01-01) est ouvert **une
seule fois**, à la fin.

## Où ajouter quoi

| Vous voulez… | Écrivez dans… |
|---|---|
| tester une nouvelle idée de trading | `alpha/families/<nom>.py` |
| ajouter un indicateur | `features/indicators.py` (source unique, jamais dupliqué) |
| ajouter un instrument | rien — gelez ses snapshots, le registre s'en charge |
| changer un seuil de protocole | `config.py` uniquement |
| ajouter une métrique | `backtest/metrics.py` |
