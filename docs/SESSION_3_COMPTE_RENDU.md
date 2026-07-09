# Session 3 — Compte-rendu : contrôle cross-asset (Nasdaq)

## Question de la session
L'échec du price action mécanique sur l'or venait-il de l'**instrument** (or très
efficient, cher) ou de l'**approche** elle-même ? Réponse par rejeu des 7 hypothèses
S1 à l'identique sur le Nasdaq (NAS100), spread réel, **aucune ré-optimisation**.

## Ce qui a été fait
- Vérification sha256 des 4 snapshots or (intacts).
- Snapshots Nasdaq gelés : H1/M15 × BID/ASK, 2015-2026, sha256, schéma aligné sur l'or.
  Instrument : `INSTRUMENT_IDX_AMERICA_E_NQ_100` (E_NQ-100).
- Couche de spread réel Nasdaq (ASK.open − BID.open), 0 négatif, grille BID==ASK exacte.
- 7 hypothèses rejouées in-sample sur **2019-2022** (repère intraday aligné sur l'or).
- Hold-out ≥ 2023 gardé vierge. Tout committé et poussé.

## Régime de séance (découverte)
Le Nasdaq CFD Dukascopy a un régime de séance changeant : ~15 barres/j H1 en 2015-2017,
transition 2018, puis ~23 barres/j dès 2019 (aligné sur l'or). Décision : snapshot complet
2015-2026 gelé (symétrique à l'or), mais **replay restreint à ≥2019** pour repère commun.
Décision gravée dans le champ `session_note` de chaque manifeste Nasdaq.

## Profil de spread (inverse de l'or)
- Or : pic ponctuel au rollover 22h (~3× la médiane).
- Nasdaq : plateau bas ~1.3 pts pendant la séance cash US (14h-21h UTC), plateau haut
  ~3.3 pts hors-US. Médiane globale ~2.5 pts (~1.95 bps).

## Résultat : tableau comparatif or vs Nasdaq (spread réel, in-sample)
| Hypothèse | R or réel | R Nasdaq réel | delta (Nas−or) |
|---|---|---|---|
| 0. cassure | −0.053 | −0.023 | +0.030 |
| A. +EMA200 | −0.060 | +0.004 | +0.064 |
| B. +structure | −0.048 | +0.054 | +0.102 |
| C. +discount | −0.204 | +0.344* | +0.548 |
| D. rejet simple | −0.127 | −0.071 | +0.056 |
| E. rejet mèche | −0.112 | −0.124 | −0.012 |
| F. M15×H1 | −0.182 | −0.047 | +0.136 |
| S. sessions 21h→23h | −0.211 | −0.123 | +0.088 |

(*) C = 39 trades, IC [−0.117, +0.807] chevauche 0 → mirage petit échantillon, rejeté.

## Verdict
**0 hypothèse avec edge**, ni sur l'or ni sur le Nasdaq (aucune ne passe : IC exclut 0
ET bat_hasard ≥ 0.95 ET R > 0). MAIS le Nasdaq est **moins mauvais sur 7/8 hypothèses**
(delta > 0), avec un gradient de contexte plus marqué (0→A→B monte plus vite que sur l'or).

Lecture : ni « c'est l'instrument » (sinon edge net sur Nasdaq — absent), ni « purement
l'approche » (sinon les deux également morts — Nasdaq décalé). **L'approche mécanique simple
est insuffisante quel que soit l'instrument, mais le Nasdaq est plus réactif au contexte
structurel.** Si un edge existe, c'est du côté du contexte (SMC réel), pas de l'instrument.

## Réserve
In-sample Nasdaq ~4 ans (2019-2022) vs ~8 ans or → IC plus larges. Le « moins mauvais »
est robuste en tendance, chiffré avec moins de précision.

## Implication règle d'arrêt
Branche (1) cross-asset faite : ne ressuscite pas « changer d'instrument », mais ne ferme
pas le projet — pointe vers la branche (2). Reste : **vraies définitions SMC, paramètres
pré-enregistrés**. Si cette branche échoue aussi → verdict « pas d'edge mécanique » final.

## Prochaine étape
Branche SMC réelle, de préférence sur Nasdaq (terrain plus réactif), avec pré-enregistrement
strict des définitions et des gates avant tout backtest.

## Livrables
- `scripts/freeze_nasdaq.py`, `scripts/spread_nasdaq.py`, `scripts/replay_nasdaq.py`,
  `scripts/replay_nasdaq_sessions.py`, `scripts/compare_or_nasdaq.py`
- `docs/SESSION_3_COMPARATIF_OR_NASDAQ.md`
- 4 snapshots Nasdaq immuables (H1/M15 × BID/ASK)

## Journal git de la session
- 066037c : gel snapshots Nasdaq + freeze_nasdaq
- 78f6254 : couche de spread réel Nasdaq
- 31a2915 : rejeu 7 hypothèses (spread réel, in-sample 2019-2022) — toutes négatives
- 57e50d5 : tableau comparatif or vs Nasdaq + livrable markdown
