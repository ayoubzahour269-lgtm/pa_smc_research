# Session 3 — Comparatif or vs Nasdaq (controle cross-asset)

Spread reel, in-sample. Or 2015-2022 (~8 ans), Nasdaq 2019-2022 (~4 ans, repere intraday aligne >=2019). Params S1 figes, aucune re-optimisation.

| Hypothese | R or reel | R Nasdaq reel | delta (Nas - or) |
|---|---|---|---|
| 0. cassure | -0.0530 | -0.0226 | +0.0304 |
| A. +EMA200 | -0.0600 | +0.0043 | +0.0643 |
| B. +structure | -0.0480 | +0.0537 | +0.1017 |
| C. +discount | -0.2040 | +0.3443 | +0.5483 |
| D. rejet simple | -0.1270 | -0.0707 | +0.0563 |
| E. rejet meche | -0.1120 | -0.1242 | -0.0122 |
| F. M15xH1 | -0.1824 | -0.0466 | +0.1358 |
| S. sessions | -0.2110 | -0.1226 | +0.0884 |

**Verdict** : 0 hypothese avec edge (R>0 + IC exclut 0 + bat_hasard>=0.95) sur aucun des deux instruments. Mais toutes les hypotheses directionnelles (0,A,B) sont MOINS negatives sur Nasdaq (delta>0), avec un gradient de contexte plus marque. Lecture : l'approche mecanique simple est insuffisante sur les deux, mais l'instrument n'est pas neutre — le Nasdaq reagit davantage au contexte structurel.
