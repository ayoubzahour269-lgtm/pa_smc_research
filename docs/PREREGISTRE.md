# Pré-enregistrement

**Gelé le 2026-08-11, avant tout backtest de la nouvelle plateforme.**

Ce document fixe les définitions, les paramètres et les critères d'acceptation **avant**
d'avoir vu le moindre résultat. Sa raison d'être est simple : sans lui, il est
impossible de distinguer une découverte d'un ajustement rétrospectif, parce qu'on peut
toujours trouver après coup une définition qui rend un résultat positif.

Toute modification ultérieure doit être ajoutée en fin de document, datée et motivée.
On n'édite pas l'historique.

---

## 1. Données et fenêtres

| Élément | Valeur gelée |
|---|---|
| Instruments disponibles | XAUUSD, NAS100 (H1 et M15, BID et ASK) |
| Instruments prévus | EURUSD, GBPUSD, USDJPY, US500 — activés dès dépôt des snapshots |
| Début in-sample XAUUSD | 2015-01-01 |
| Début in-sample NAS100 | 2019-01-01 (changement de régime de séance du flux avant cette date) |
| Fin in-sample / début hold-out | **2023-01-01** |
| Coût d'exécution | spread réel `ASK.open − BID.open`, barre par barre |
| Repli si spread absent | 2 % de 1R, volontairement pénalisant |

**Règle du hold-out.** Les données ≥ 2023-01-01 ne sont lues qu'**une seule fois**, à la
fin du projet, après que tout le reste est figé. Ce qui y est mesuré est le résultat,
définitivement. Aucun retour en arrière pour ajuster : un hold-out consulté deux fois ne
vaut plus rien.

---

## 2. Exécution

| Règle | Valeur |
|---|---|
| Entrée | ouverture de la barre **suivant** le signal |
| Coût | spread de la barre d'entrée, prélevé intégralement à l'entrée |
| Barre où stop et objectif sont tous deux atteignables | **le stop l'emporte** |
| Taux de barres ambiguës | compté et publié systématiquement |
| Sortie à l'échéance | clôture de la dernière barre détenue |
| `max_hold` | nombre de barres détenues, entrée comprise |

L'hypothèse intrabarre défavorable n'est pas du pessimisme : c'est la seule qui ne
fabrique pas d'edge en l'absence de données tick.

---

## 3. Définitions des features

Toutes causales : la valeur en `t` n'utilise que les barres `≤ t`. Vérifié
mécaniquement par `tests/antilookahead/`, sur données synthétiques **et** réelles.

### 3.1 Indicateurs

| Feature | Définition gelée |
|---|---|
| True range | `max(h−l, |h−c₋₁|, |l−c₋₁|)` |
| ATR | lissage de Wilder (`alpha = 1/n`), `n = 14` par défaut |
| EMA | `adjust=False` (pas de repondération rétroactive) |
| RSI | Wilder, `n = 14` ; vaut 100 si la perte moyenne est nulle |
| ADX | Wilder, `n = 14` |
| `rolling_high(n)` | plus haut des `n` barres **précédentes**, barre courante exclue |
| `atr_ratio(f, s)` | ATR(5) / ATR(50) — < 1 compression, > 1 expansion |
| VWAP | ancré sur la journée de trading ; poids uniformes si volume ≤ 0 |

### 3.2 Séances (heures UTC, fin exclue)

| Fenêtre | Bornes |
|---|---|
| Asie | 00–07 |
| Londres | 07–13 |
| New York | 13–21 |
| Rollover | 21–24 |
| Killzone Londres | 07–10 |
| Killzone New York | 12–15 |
| Killzone clôture NY | 19–21 |

Bascule de journée de trading : **22h UTC** (rollover des courtiers FX/CFD, et pic de
spread mesuré sur l'or à 3,2× la médiane).

Le range d'une séance n'est exposé qu'**après la clôture** de cette occurrence. Les
occurrences sont des séquences contiguës de barres, pas des groupes par journée — une
fenêtre franchissant la bascule formerait sinon un groupe discontinu qui laisserait
fuiter le futur.

### 3.3 Structure de prix (SMC)

| Concept | Définition mécanique gelée |
|---|---|
| Pivot haut | plus-haut de la fenêtre centrée `[k−W, k+W]`, `W = 3` par défaut |
| **Confirmation** | un pivot en `k` n'est exposé qu'à partir de `k+W` |
| Structure haussière | clôture > dernier pivot haut confirmé |
| **BOS** | cassure dans le sens de la structure déjà établie (continuation) |
| **CHoCH** | première cassure contraire à la structure en cours (retournement) |
| Équilibre | milieu de la dernière jambe confirmée (haut + bas)/2 |
| Discount / Premium | position < 0,5 / > 0,5 dans la jambe |
| OTE | retracement 0,618–0,79 |
| FVG haussier | `low[i] > high[i−2]`, taille ≥ 0,25 × ATR |
| FVG baissier | `high[i] < low[i−2]`, taille ≥ 0,25 × ATR |
| Impulsion | déplacement ≥ 1,5 × ATR sur ≤ 3 barres |
| Order block | dernière bougie de sens opposé dans la fenêtre d'impulsion précédant une cassure |
| Egal high/low | extrême des 20 barres précédentes, tolérance 0,1 × ATR |
| **Balayage** | mèche traverse le niveau **et** clôture revient en deçà, mèche ≥ 0,1 × ATR |
| **Cassure** | traverse **et** clôture au-delà — explicitement distinguée du balayage |

Le décalage de confirmation des pivots est la protection centrale du module. Exposer un
pivot à sa date de formation suffit à produire une courbe d'équité magnifique et
entièrement fausse.

---

## 4. Critères d'acceptation (gate)

Une hypothèse n'a le droit d'être qualifiée d'« intéressante » que si **toutes** les
portes sont franchies :

| # | Porte | Seuil gelé |
|---|---|---|
| 1 | Taille d'échantillon | ≥ 100 trades |
| 2 | Espérance-R au spread réel | > 0 |
| 3 | IC bootstrap 95 % (10 000 tirages) | exclut 0 |
| 4 | Témoin aléatoire apparié | battu dans ≥ 95 % des cas, 200 témoins |
| 5 | Sensibilité au coût | R > 0 encore à **1,5× le spread observé** |
| 6 | Anti-anticipation | suite `tests/antilookahead/` verte |
| 7 | Correction pour tests multiples | significatif après Benjamini-Hochberg sur `m` = nombre total d'essais enregistrés |

**Appariement du témoin** : même symbole, même sens, même géométrie de risque (stop,
objectif, échéance) et **même heure UTC**. Seul le choix du moment est randomisé.

L'appariement horaire n'est pas un raffinement. Sans lui, une « stratégie » qui se
contente d'éviter les heures à spread élevé bat le hasard et serait publiée comme une
découverte, alors qu'elle ne contient aucune information sur la direction du prix. Un
test dédié (`test_le_temoin_horaire_demasque_un_artefact_de_creneau`) vérifie que le
protocole rejette bien ce cas.

---

## 5. Comptage des essais

Chaque configuration testée est enregistrée dans `docs/journal_essais.jsonl`
(append-only) **avant** que son résultat ne soit lu. Un essai = un couple (famille,
jeu de paramètres, univers, timeframe, fenêtre).

Motif : à 5 % de seuil, 100 configurations testées produisent mécaniquement ~5
« stratégies gagnantes » qui ne sont que du bruit. Toute correction pour tests multiples
est cosmétique si l'on ne connaît pas le nombre réel d'essais — et ce nombre ne peut pas
être reconstitué a posteriori, parce qu'on oublie les essais ratés.

---

## 6. Familles explorées

Aucune famille n'a de statut privilégié. SMC est la n° 9, jugée aux mêmes critères que
les autres.

1. Effets de séance / heure
2. Opening range breakout
3. Retour à la moyenne intraday
4. Momentum conditionné au régime
5. Compression → expansion de volatilité
6. Gap et dérive overnight
7. Lead-lag inter-actifs
8. Divergence de corrélation / pairs
9. SMC / structure de prix
10. Microstructure et liquidité
11. Événements macro
12. Saisonnalité calendaire
13. Modèle supervisé sur l'union des features
14. Ensemble des familles survivantes

---

## 7. Porte de faisabilité du scalping M1/M5

Avant toute stratégie sur M1 ou M5, on mesure :

- `spread médian ÷ range vrai médian de la barre` ;
- `spread médian ÷ excursion favorable médiane` sur la durée de détention visée.

**Seuil go/no-go gelé : si le coût dépasse 25 % de l'excursion favorable médiane, le
timeframe est déclaré non tradable** et aucune stratégie n'y sera développée.

Repères existants : spread médian 0,33 $ sur l'or H1 (1,05 $ au rollover de 22h),
~2,5 points sur le Nasdaq. Rapportés à une barre M1, ces coûts changent d'ordre de
grandeur relatif.

---

## 8. Ce qui n'est PAS repris de l'historique du dépôt

Les paramètres des sessions précédentes (`N_BREAK = 20`, `ATR = 14` en moyenne simple,
SL 1,5× / TP 3,0×, les sept hypothèses de la Session 1) ne servent pas de référence. Ils
ont été choisis dans un autre cadre et leurs résultats sont connus — les réutiliser
reviendrait à poursuivre une recherche déjà close plutôt qu'à en ouvrir une nouvelle.

Ce qui est conservé de cet historique : les données, le hold-out vierge, et la
connaissance des huit hypothèses déjà rejetées — savoir ce qui a échoué évite de le
retester sans le savoir.

---

## Journal des modifications

_(aucune à ce jour)_
