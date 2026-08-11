# Exploration H1 — NAS100, XAUUSD

Fenetre in-sample, cout d'execution reel (spread ASK−BID barre par barre). Hold-out (≥ 2023-01-01) non ouvert.

| famille | symbole | trades | R_moyen | IC95 | taux_reussite | R_cout_majore | bat_hasard | p_value | q_value | gate | survit_correction |
|---|---|---|---|---|---|---|---|---|---|---|---|
| smc_bos | NAS100 | 348 | 0.1293 | (-0.0218, 0.2848) | 0.3908 | 0.1082 | 0.975 | 0.029850746268656716 | 0.3681592039800995 | False | False |
| cassure[compression] | NAS100 | 146 | 0.113 | (-0.1202, 0.3589) | 0.3904 | 0.0839 | 0.885 | 0.11940298507462686 | 0.5522388059701493 | False | False |
| cassure[aucun]/suiveur1.0R | NAS100 | 814 | 0.1047 | (0.0288, 0.1796) | 0.4496 | 0.086 | 0.995 | 0.009950248756218905 | 0.18407960199004975 | True | False |
| cassure[aucun] | NAS100 | 778 | 0.0966 | (-0.0036, 0.1971) | 0.3792 | 0.078 | 0.995 | 0.009950248756218905 | 0.18407960199004975 | False | False |
| opening_range | NAS100 | 932 | 0.0933 | (0.0014, 0.1849) | 0.383 | 0.0751 | 0.99 | 0.014925373134328358 | 0.2208955223880597 | True | False |
| cassure[spread_bas] | NAS100 | 618 | 0.0751 | (-0.0393, 0.188) | 0.3706 | 0.0587 | 0.95 | 0.05472636815920398 | 0.40497512437810945 | False | False |
| cassure[aucun]/cloture21h | NAS100 | 784 | 0.0711 | (-0.0257, 0.1685) | 0.3916 | 0.0524 | 0.995 | 0.009950248756218905 | 0.18407960199004975 | False | False |
| cassure[aucun]/seuil1.0R | NAS100 | 790 | 0.0677 | (-0.0226, 0.156) | 0.2848 | 0.0491 | 0.995 | 0.009950248756218905 | 0.18407960199004975 | False | False |
| cassure[compression] | XAUUSD | 64 | 0.0562 | (-0.3136, 0.4265) | 0.3906 | -0.0016 | 0.77 | 0.23383084577114427 | 0.7787983161117489 | False | False |
| gap_ouverture | XAUUSD | 170 | 0.0507 | (-0.1611, 0.2642) | 0.4118 | -0.014 | 0.96 | 0.04477611940298507 | 0.3681592039800995 | False | False |
| momentum_series_temporelles | NAS100 | 965 | 0.0477 | (-0.0426, 0.1377) | 0.3617 | 0.0308 | 0.695 | 0.30845771144278605 | 0.8152096659559346 | False | False |
| drift_overnight[achat] | NAS100 | 982 | -0.0043 | (-0.0946, 0.0829) | 0.3615 | -0.0376 | 0.515 | 0.48756218905472637 | 0.9028666192845297 | False | False |
| divergence_correlation[XAUUSD] | NAS100 | 204 | -0.0094 | (-0.2057, 0.1881) | 0.3529 | -0.041 | 0.74 | 0.263681592039801 | 0.7787983161117489 | False | False |
| smc_bos | XAUUSD | 498 | -0.0114 | (-0.1354, 0.115) | 0.3675 | -0.0622 | 0.94 | 0.06467661691542288 | 0.4350972410673903 | False | False |
| cassure[adx] | NAS100 | 404 | -0.0168 | (-0.1518, 0.1206) | 0.3416 | -0.0348 | 0.605 | 0.39800995024875624 | 0.8879133743049459 | False | False |
| repli_de_tendance | NAS100 | 387 | -0.0183 | (-0.158, 0.1196) | 0.3488 | -0.0362 | 0.46 | 0.5422885572139303 | 0.9203980099502487 | False | False |
| cassure[spread_bas] | XAUUSD | 523 | -0.0201 | (-0.1379, 0.1013) | 0.3633 | -0.0645 | 0.91 | 0.0945273631840796 | 0.5380788365862993 | False | False |
| niveaux_ronds[rejet] | NAS100 | 2649 | -0.0214 | (-0.0741, 0.0329) | 0.3499 | -0.0535 | 0.915 | 0.08955223880597014 | 0.5380788365862993 | False | False |
| ensemble | NAS100 | 280 | -0.0222 | (-0.1895, 0.1446) | 0.3464 | -0.0454 | 0.595 | 0.4079601990049751 | 0.8879133743049459 | False | False |
| divergence_correlation[NAS100] | XAUUSD | 194 | -0.0322 | (-0.2282, 0.167) | 0.366 | -0.0768 | 0.805 | 0.19900497512437812 | 0.753849798625918 | False | False |
| fade_range_asiatique | NAS100 | 778 | -0.0456 | (-0.1472, 0.055) | 0.3548 | -0.0999 | 0.965 | 0.03980099502487562 | 0.3681592039800995 | False | False |
| niveaux_ronds[cassure] | NAS100 | 2627 | -0.0456 | (-0.0992, 0.0088) | 0.343 | -0.0786 | 0.73 | 0.2736318407960199 | 0.7787983161117489 | False | False |
| lead_lag[XAUUSD]+ | NAS100 | 1128 | -0.049 | (-0.13, 0.0331) | 0.3422 | -0.0824 | 0.57 | 0.43283582089552236 | 0.8897180762852405 | False | False |
| cassure_ratee | XAUUSD | 1673 | -0.0522 | (-0.1196, 0.0156) | 0.3556 | -0.1028 | 0.96 | 0.04477611940298507 | 0.3681592039800995 | False | False |
| lead_lag[XAUUSD]- | NAS100 | 1114 | -0.0608 | (-0.1432, 0.0227) | 0.3357 | -0.0942 | 0.575 | 0.42786069651741293 | 0.8897180762852405 | False | False |
| saisonnalite_intraday | NAS100 | 2215 | -0.0647 | (-0.1219, -0.0055) | 0.3314 | -0.088 | 0.34 | 0.6616915422885572 | 0.9487179487179487 | False | False |
| lead_lag[NAS100]+ | XAUUSD | 1098 | -0.0659 | (-0.1472, 0.0183) | 0.3452 | -0.1102 | 0.885 | 0.11940298507462686 | 0.5522388059701493 | False | False |
| regime_volatilite | NAS100 | 1410 | -0.0669 | (-0.1392, 0.0069) | 0.334 | -0.0983 | 0.505 | 0.4975124378109453 | 0.9028666192845297 | False | False |
| smc_fvg | NAS100 | 991 | -0.0687 | (-0.1553, 0.0197) | 0.336 | -0.1024 | 0.435 | 0.5671641791044776 | 0.9277611940298507 | False | False |
| momentum_intraseance | NAS100 | 812 | -0.0699 | (-0.1661, 0.0282) | 0.3387 | -0.1032 | 0.675 | 0.3283582089552239 | 0.8194511314395763 | False | False |
| microstructure_choc | XAUUSD | 1111 | -0.073 | (-0.1569, 0.0095) | 0.3519 | -0.1309 | 0.9 | 0.1044776119402985 | 0.5522388059701492 | False | False |
| fin_de_mois | XAUUSD | 331 | -0.0734 | (-0.2224, 0.081) | 0.3444 | -0.1218 | 0.79 | 0.21393034825870647 | 0.753849798625918 | False | False |
| fade_range_asiatique | XAUUSD | 1744 | -0.0752 | (-0.1412, -0.0095) | 0.3446 | -0.1256 | 0.82 | 0.18407960199004975 | 0.753849798625918 | False | False |
| cassure[aucun]/seuil1.0R | XAUUSD | 1031 | -0.0791 | (-0.1538, -0.0026) | 0.2619 | -0.1247 | 0.795 | 0.208955223880597 | 0.753849798625918 | False | False |
| retour_vwap | NAS100 | 640 | -0.0798 | (-0.1875, 0.0278) | 0.3219 | -0.0978 | 0.145 | 0.8557213930348259 | 1.0 | False | False |
| gap_ouverture | NAS100 | 213 | -0.081 | (-0.2709, 0.1115) | 0.338 | -0.1247 | 0.66 | 0.34328358208955223 | 0.8194511314395763 | False | False |
| niveaux_ronds[cassure] | XAUUSD | 2963 | -0.0846 | (-0.1348, -0.0339) | 0.3432 | -0.131 | 0.875 | 0.12935323383084577 | 0.5630670178519168 | False | False |
| cassure[aucun]/suiveur1.0R | XAUUSD | 1046 | -0.0846 | (-0.1445, -0.0214) | 0.3642 | -0.1309 | 0.745 | 0.25870646766169153 | 0.7787983161117489 | False | False |
| cassure[aucun]/cloture21h | XAUUSD | 1026 | -0.0849 | (-0.1632, -0.004) | 0.3616 | -0.1305 | 0.745 | 0.25870646766169153 | 0.7787983161117489 | False | False |
| cassure[aucun] | XAUUSD | 1024 | -0.0893 | (-0.1748, -0.0007) | 0.3398 | -0.1349 | 0.705 | 0.29850746268656714 | 0.8152096659559346 | False | False |
| cassure_ratee | NAS100 | 953 | -0.1 | (-0.1893, -0.0098) | 0.3263 | -0.1344 | 0.335 | 0.6666666666666666 | 0.9487179487179487 | False | False |
| seance_derive | XAUUSD | 2035 | -0.1042 | (-0.1653, -0.0414) | 0.3332 | -0.1524 | 0.49 | 0.5124378109452736 | 0.9028666192845297 | False | False |
| regime_volatilite | XAUUSD | 2806 | -0.105 | (-0.1572, -0.0526) | 0.3361 | -0.1559 | 0.595 | 0.4079601990049751 | 0.8879133743049459 | False | False |
| niveaux_ronds[rejet] | XAUUSD | 3056 | -0.1136 | (-0.1617, -0.0647) | 0.3315 | -0.16 | 0.455 | 0.5472636815920398 | 0.9203980099502487 | False | False |
| smc_balayage | NAS100 | 666 | -0.1194 | (-0.2252, -0.0128) | 0.3108 | -0.1384 | 0.02 | 0.9800995024875622 | 1.0 | False | False |
| smc_balayage | XAUUSD | 970 | -0.1217 | (-0.2087, -0.0344) | 0.333 | -0.1678 | 0.42 | 0.582089552238806 | 0.9277611940298507 | False | False |
| opening_range | XAUUSD | 1432 | -0.1262 | (-0.1981, -0.0533) | 0.3296 | -0.1751 | 0.38 | 0.6218905472636815 | 0.9277611940298507 | False | False |
| momentum_multijour | XAUUSD | 618 | -0.1296 | (-0.2384, -0.0179) | 0.3236 | -0.1707 | 0.38 | 0.6218905472636815 | 0.9277611940298507 | False | False |
| cassure[adx] | XAUUSD | 551 | -0.1307 | (-0.2449, -0.0149) | 0.3249 | -0.1764 | 0.4 | 0.6019900497512438 | 0.9277611940298507 | False | False |
| smc_choch | NAS100 | 277 | -0.1341 | (-0.2963, 0.0288) | 0.3141 | -0.1629 | 0.21 | 0.7910447761194029 | 1.0 | False | False |
| momentum_multijour | NAS100 | 316 | -0.1389 | (-0.2863, 0.0135) | 0.2975 | -0.1526 | 0.035 | 0.9651741293532339 | 1.0 | False | False |
| retour_vwap | XAUUSD | 999 | -0.1425 | (-0.2266, -0.0557) | 0.3213 | -0.1894 | 0.22 | 0.7810945273631841 | 1.0 | False | False |
| lead_lag[NAS100]- | XAUUSD | 1128 | -0.1446 | (-0.2241, -0.0647) | 0.3165 | -0.1887 | 0.24 | 0.7611940298507462 | 1.0 | False | False |
| momentum_series_temporelles | XAUUSD | 1961 | -0.1468 | (-0.2054, -0.086) | 0.3202 | -0.1904 | 0.15 | 0.8507462686567164 | 1.0 | False | False |
| drift_overnight[achat] | XAUUSD | 1960 | -0.1476 | (-0.2105, -0.085) | 0.3684 | -0.2665 | 0.51 | 0.4925373134328358 | 0.9028666192845297 | False | False |
| smc_fvg | XAUUSD | 1732 | -0.1496 | (-0.2129, -0.0848) | 0.3222 | -0.2002 | 0.105 | 0.8955223880597015 | 1.0 | False | False |
| fin_de_mois | NAS100 | 169 | -0.1515 | (-0.3582, 0.0616) | 0.3195 | -0.2065 | 0.235 | 0.7661691542288557 | 1.0 | False | False |
| ensemble | XAUUSD | 388 | -0.1692 | (-0.305, -0.0305) | 0.3144 | -0.2189 | 0.24 | 0.7611940298507462 | 1.0 | False | False |
| smc_choch | XAUUSD | 496 | -0.1816 | (-0.3025, -0.0619) | 0.3165 | -0.2333 | 0.125 | 0.8756218905472637 | 1.0 | False | False |
| seance_derive | NAS100 | 1029 | -0.1871 | (-0.2707, -0.1025) | 0.309 | -0.2423 | 0.155 | 0.845771144278607 | 1.0 | False | False |
| drift_overnight[vente] | NAS100 | 991 | -0.188 | (-0.2722, -0.1014) | 0.2957 | -0.2215 | 0.51 | 0.4925373134328358 | 0.9028666192845297 | False | False |
| saisonnalite_intraday | XAUUSD | 3430 | -0.2296 | (-0.2758, -0.1824) | 0.3087 | -0.301 | 0.0 | 1.0 | 1.0 | False | False |
| repli_de_tendance | XAUUSD | 680 | -0.2305 | (-0.333, -0.1295) | 0.3 | -0.28 | 0.035 | 0.9651741293532339 | 1.0 | False | False |
| momentum_intraseance | XAUUSD | 1438 | -0.2419 | (-0.3144, -0.1669) | 0.3345 | -0.3586 | 0.67 | 0.3333333333333333 | 0.8194511314395763 | False | False |
| microstructure_choc | NAS100 | 12 | -0.2896 | (-1.0297, 0.4603) | 0.25 | -0.3094 | 0.375 | 0.6268656716417911 | 0.9277611940298507 | False | False |
| drift_overnight[vente] | XAUUSD | 1963 | -0.3636 | (-0.4233, -0.304) | 0.2965 | -0.4824 | 0.49 | 0.5124378109452736 | 0.9028666192845297 | False | False |
| cointegration[XAUUSD] | NAS100 | 23 | -0.4443 | (-0.8838, 0.0905) | 0.2174 | -0.4925 | 0.1 | 0.900497512437811 | 1.0 | False | False |
| cointegration[NAS100] | XAUUSD | 26 | -0.4448 | (-0.8962, 0.0505) | 0.2308 | -0.5012 | 0.08 | 0.9203980099502488 | 1.0 | False | False |

## Verdict

AUCUNE famille retenue sur 68 evaluees (74 essais journalises, FDR = 0.05).
Detail de la correction seule : Aucune hypothese ne survit a la correction (m = 74 essais, FDR = 0.05). Les esperances positives observees sont compatibles avec le hasard.
34 familles declarees, 27.0 tests effectivement independants (redondance 21%).
Groupes de familles redondantes :
  - cassure[adx], cassure[aucun], cassure[aucun]/cloture21h, cassure[aucun]/seuil1.0R, cassure[aucun]/suiveur1.0R, cassure[compression], cassure[spread_bas], opening_range, smc_bos
  - cassure_ratee, smc_balayage
Borne basse (correction sur les tests EFFECTIVEMENT independants, m = 68) : 0 survivante(s). Cette borne n'est PAS le critere de retenue — elle indique seulement si le rejet tient a la severite du comptage ou a la faiblesse de la preuve.
Facteur dollar INDISPONIBLE : EURUSD, GBPUSD, USDJPY absents du disque. Les familles inter-actifs n'ont donc ete testees que sur l'axe risk-on/risk-off, pas sur le dollar.

## Reserves

- Resultats in-sample : ils ne disent rien de la performance future.
- Taux de barres ambigues le plus eleve : 4.3%.
- Une q-value faible ne suffit pas : la correction ne juge que la p-value du temoin, pas le signe de l'esperance. Seule la colonne `survit_correction` combine les deux.