# Session 2 — Compte-rendu : couche de spread historique reel

**Date :** 2026-07-08
**Depot :** pa_smc_research | **Portee :** Projet A (recherche offline).
Aucune decision de capital. Hold-out (>= 2023-01-01) reste vierge.

## Objectif

Remplacer le cout d'execution estime (forfait 0,40 / 0,80 $) de la S1 par une couche de
spread historique reel (ASK - BID, barre par barre), puis rejouer les hypotheses au cout reel.
Angle mort de la S1 : le seul signal apparemment reel (sessions 21h->23h) avait ete detruit
precisement par le cout.

## Livrables

### 1. Snapshots ASK immuables (symetriques aux BID)

| Timeframe / Side | Barres | sha256 |
|---|---|---|
| H1 BID  | 68041  | `1c9f7d0349f7` |
| H1 ASK  | 68041  | `f3383a6ca28e` |
| M15 BID | 271660 | `410784c4c325` |
| M15 ASK | 271660 | `847e18fb2653` |

Meme grille horodatee que les BID (compte au bar pres, trous au trou pres), ASK partout > BID.

### 2. Couche de spread reel + rapport horaire

- Spread = ASK.open - BID.open (mesure sur l'OPEN car le moteur entre a o[i+1]).
- Alignement 1:1 sur la grille BID (aucun NaN), signe propre (0 negatif sur 11 ans).
- Spread global H1 : mediane 0,331 $ | moyenne 0,42 $ | p95 0,85 $ | p99 1,87 $ | max 14,9 $.
- Pic rollover a 22h UTC : mediane 1,054 $ (3,2x la mediane globale), p95 3,16 $.
  Plateau bas et stable (~0,33 $) sur les 22 autres heures.

### 3. Moteur etendu (run_backtest)

- Nouveau parametre optionnel cost_series : cout du trade = spread reel a l'entree (i+1),
  repli conservateur sur cost_usd si NaN.
- Forfait cost_usd conserve comme defaut (non-regression ; moteur partage A/B).
- Tests : 5 passed (2 non-regression S1 + 3 nouveaux couvrant cost_series).

### 4. Verdict au cout reel (in-sample, hold-out intouche)

| Hypothese | R estime (forfait 0,40) | R reel (spread) | Statut |
|---|---|---|---|
| 0. cassure | -0,074 | -0,053 | perdante |
| A. +EMA200 | -0,081 | -0,060 | perdante |
| B. +structure | -0,069 | -0,048 | perdante |
| C. +discount | -0,230 | -0,204 | perdante |
| D. rejet simple | -0,145 | -0,127 | perdante |
| E. rejet meche | -0,128 | -0,112 | perdante |
| S. sessions 21h->23h | +0,061 | -0,211 | disqualifiee (mirage rollover) |

Cout reel session (spread 21h + 23h, par jour) : mediane 1,16 $ (~3x le forfait normal).

## Apprentissages cles

1. Le spread reel est heterogene et decisif. Mediane 0,33 $ en heures liquides, explosion au
   rollover 22h. Un forfait plat masque DEUX erreurs opposees.
2. Le forfait n'etait pas faux uniformement : trop pessimiste pour les cassures (heures liquides,
   delta_R ~+0,02), massivement trop optimiste pour les sessions (rollover).
3. Le signal "sessions" etait un mirage de liquidite : +0,19 R brut, +0,06 R au forfait 0,40
   (IC excluant zero), mais -0,21 R au cout reel. Raison d'etre de cette session.
4. Gouvernance moteur : forfait garde par defaut, spread reel via cost_series.
   Regle machine : toujours activer le .venv DU DEPOT, jamais ~/.venv.

## Conclusion

Les 7 hypotheses testees sont negatives au cout reel. Aucun edge exploitable en price action
mecanique simple sur l'or. Le "non" de la S1 est confirme et durci. L'infrastructure de mesure
(spread reel + moteur etendu + tests) est le veritable acquis : elle a fait tomber un faux
positif que le cout estime laissait passer.

## Journal git de la session

```
d174c04 Session 2 - Etape 4 : rejeu 7 hypotheses au cout reel (in-sample) - toutes negatives, signal sessions +0.06R forfait -> -0.21R reel (mirage rollover)
158c9f0 Session 2 - Etape 3 : moteur accepte cost_series (spread reel/barre), forfait garde par defaut + 3 tests (5 passed)
4587466 Session 2 - Etape 2 : couche de spread ASK-BID (H1+M15) + rapport par heure UTC (pic rollover 22h)
d03b8d8 Session 2 - Etape 1 : snapshots immuables XAUUSD ASK H1+M15 2015-2026 v1 (sha256, symetriques au BID)
79782ca Etape 7 : multi-timeframe H1xM15 rejete - perdant des l'in-sample
```
