# alphalab

Plateforme de recherche et de production de plans de trading intraday.

Objectif : produire **chaque jour** un plan de positions gradé, en combinant analyse
technique, structure de prix, corrélations inter-actifs et contexte macro — et en
**mesurant** la probabilité de chaque candidat au lieu de l'affirmer.

## Pourquoi cette plateforme est construite ainsi

Ce dépôt a un historique : trois sessions de recherche y ont testé huit hypothèses de
price action mécanique sur deux instruments. **Toutes ont été rejetées** une fois le
coût d'exécution réel appliqué. Le seul signal qui avait semblé vivant (créneau
21h→23h, +0,06R au coût forfaitaire) s'est révélé perdant à −0,21R au spread réel :
le pic de rollover de 22h le mangeait entièrement.

Trois principes en découlent, et ils sont câblés dans le code, pas seulement écrits ici :

1. **Le coût est mesuré, jamais supposé.** Le spread réel (`ASK.open − BID.open`,
   barre par barre) est le mode par défaut. Si un instrument n'a pas ses deux côtés,
   la couche de coût lève une erreur au lieu d'inventer un forfait.
2. **Une espérance positive ne prouve rien seule.** Toute hypothèse doit battre un
   témoin aléatoire apparié en densité, biais acheteur/vendeur, géométrie de risque
   **et heure de la journée**. Sans l'appariement horaire, une « stratégie » qui ne
   fait qu'éviter les heures chères passerait pour une découverte.
3. **Explorer largement multiplie les faux positifs.** À 5 % de seuil, 100
   configurations testées en produisent ~5 par pur hasard. Chaque essai est donc
   enregistré dans un journal append-only *avant* d'en lire le résultat, pour rendre
   possible une correction pour tests multiples.

Le hold-out (≥ 2023-01-01) est **vierge** et ne sera ouvert qu'une seule fois, à la fin
du projet.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

L'extra `fetch` (téléchargement Dukascopy) est séparé, car il exige un accès réseau que
l'environnement de développement n'a pas :

```bash
pip install -e ".[fetch]"
```

## Utilisation

```bash
alphalab status                  # ce qui est disponible sur disque, fenêtres in-sample
alphalab spread XAUUSD --tf H1   # profil de coût réel, par heure UTC
alphalab scalping                # durée de détention minimale viable après spread
alphalab explore --tf H1         # teste toutes les familles, rend le verdict corrigé
alphalab daily --date 2022-06-15 # plan gradé A/B/C d'une journée
alphalab freeze EURUSD --tf M15 --side BID,ASK   # télécharge et gèle (réseau requis)
```

### Le plan quotidien

Un plan est émis **chaque jour**, sans exception. Ce qui varie n'est pas son existence
mais la taille qu'il recommande :

| Grade | Condition | Risque |
|---|---|---|
| A | `E[R] ≥ +0,15` et meilleure occasion du jour | 1,0 % |
| B | `E[R] ≥ 0` | 0,5 % |
| C | `E[R] < 0` — meilleur candidat malgré tout | 0,1 % |

L'espérance vient d'une probabilité **calibrée hors échantillon**, pas d'un score de
confluence. Quand le modèle n'apporte rien de mesurable, le plan l'écrit noir sur blanc
au lieu d'afficher une probabilité rassurante. Il signale aussi les entrées tombant dans
une fenêtre structurellement chère — une entrée à 22h UTC sur l'or paie 3,2× le spread
médian, ce qui suffit à retourner le signe de l'espérance.

### Ajouter un instrument

Aucun code à modifier. Gelez ses snapshots, committez-les, c'est tout — le registre les
découvre et les familles inter-actifs s'activent :

```bash
alphalab freeze EURUSD --tf M15,H1 --side BID,ASK
alphalab freeze USDJPY --tf M15,H1 --side BID,ASK
alphalab status
```

Le **facteur dollar** (la corrélation or ↔ EUR/USD ↔ dollar) exige EURUSD, idéalement
avec GBPUSD et USDJPY. Tant qu'ils sont absents, il est déclaré **indisponible** dans les
rapports plutôt qu'approximé par un proxy trompeur.

## Données

`data/snapshots/` contient des snapshots **immuables**, vérifiés par sha256 contre leur
manifeste à chaque lecture. Un snapshot ne se réécrit jamais : toute nouvelle donnée est
une nouvelle version. C'est ce qui permet de comparer deux résultats à des mois
d'intervalle en sachant que l'écart vient du code, pas de la donnée.

Actuellement gelés : XAUUSD et NAS100, en H1 et M15, côtés BID et ASK, 2015→2026.

## Tests

```bash
pytest -m "not slow"     # unitaires + golden, sur fixtures synthétiques, aucune donnée requise
pytest -m slow           # intégrité sha256 des snapshots réels (~86 Mo)
ruff check . && mypy src
```

Les tests **golden** valident le moteur contre des cas dont le résultat se calcule à la
main — pas contre une implémentation antérieure.

## Structure

| Chemin | Rôle |
|---|---|
| `src/alphalab/data/` | contrat OHLCV, snapshots immuables, registre, coûts réels |
| `src/alphalab/features/` | indicateurs, séances, microstructure, SMC, inter-actifs |
| `src/alphalab/alpha/` | familles d'hypothèses → candidats |
| `src/alphalab/model/` | labellisation, walk-forward purgé, calibration, tests multiples |
| `src/alphalab/backtest/` | moteur portefeuille, métriques, protocole d'acceptation |
| `src/alphalab/risk/` | dimensionnement, plafonds ajustés de la corrélation |
| `src/alphalab/report/` | plan quotidien gradé, rapports de recherche |
| `docs/` | pré-enregistrements, journal des essais, comptes-rendus de session |
| `notebooks/legacy/` | archive figée des sessions 1 et 2 (non maintenue) |

Voir `docs/ARCHITECTURE.md` pour le flux de données détaillé.

## Avertissement

Ce dépôt est un outil de **recherche**. Il ne passe aucun ordre, ne se connecte à aucun
courtier, et ne constitue pas un conseil en investissement. Un plan de trading qu'il
produit peut afficher une espérance négative : c'est une information, pas un défaut.
