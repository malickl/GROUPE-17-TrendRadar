# Architecture du projet

Ce document explique comment le projet est rangé et comment les fichiers s'enchaînent, depuis les données Wikipédia jusqu'à l'API.

## Vue d'ensemble

```
trend-radar/
│
├── data/       Les données d'entrée
├── src/        Le code
├── models/     Le modèle entraîné et ses rapports
├── outputs/    Les résultats des scans
├── web/        La petite interface bulles (/ui)
├── tests/      Les tests de l'API
├── docs/       La documentation (API, architecture)
├── README.md, CLAUDE.md, requirements.txt
│
└── myenv/, cache/, __pycache__/, .pytest_cache/   Techniques, ignorés par Git
```

## Comment tout s'enchaîne

```
data/train_v1_t0_plus_4mois.csv ──► train.py ──► models/model_v1.joblib
                                                         │
Wikipédia (vues mensuelles) ──► scan.py ────────────────►├──► outputs/scan_AAAAMM.csv
                                                         │            │
                                                         │         demo.py (+ data/demo_contenu.json)
                                                         │            │
                                                         │            ▼
                                                         │     data/demo_scan.json
                                                         │            │
                                                      api.py ◄────────┘
                                                         │
                                               équipe tech (/scan/latest, /predict…)
```

`features.py` est utilisé par `train.py`, `scan.py` et `api.py` : c'est la seule définition du décollage et des variables (voir la règle d'or plus bas).

## `data/` : les entrées

| Fichier | Rôle |
|---|---|
| `train_v1_t0_plus_4mois.csv` | Les 175 cas étiquetés (Buzz / Tendance durable), décollés entre 2020 et 2023. Base d'entraînement. |
| `wiki_series_2018_2026.tsv` | Vues mensuelles de 2 078 pages Wikipédia (2018-01 → 2026-10). Sert de liste de pages à scanner et aux tests hors ligne. |
| `exclusions.txt` | 97 pages hors sujet ou à problème, retirées à la main. |
| `reference_30.csv` | Les 30 cas de référence de la validation. |
| `demo_contenu.json` | Textes fictifs rédigés à la main (élément clé, veille, matières premières). |
| `demo_scan.json` | Fichier de démo servi par `/scan/latest` : vrai scan + champs fictifs. |

**D'où viennent les pages Wikipédia :** on a pris 9 catégories (Desserts, Non-alcoholic drinks, Flavors, Sugar substitutes, Spices, Edible nuts and seeds, Tropical fruit, Perfume ingredients, Essential oils) et récupéré automatiquement toutes leurs pages, plus celles des sous-catégories directes (un seul niveau). Moins les exclusions : 1 981 pages scannées.

## `src/` : le code

| Fichier | Rôle | Commande |
|---|---|---|
| `features.py` | Calcul commun : moyenne 3 mois, règle de décollage (t0), 9 variables, garde-fous. | `python src/features.py` (vérifie `175/175`) |
| `train.py` | Entraîne le modèle (Random Forest) et écrit le rapport. | `python src/train.py` |
| `validate.py` | Teste le modèle sur 30 cas de référence. | `python src/validate.py` |
| `scan.py` | Cherche les pages qui décollent et les prédit. | `python src/scan.py` |
| `demo.py` | Fabrique `demo_scan.json` (champs fictifs, graine fixe). | `python src/demo.py` |
| `api.py` | L'API FastAPI appelée par l'équipe tech. | `uvicorn src.api:app` |

## `models/` : le modèle

| Fichier | Rôle |
|---|---|
| `model_v1.joblib` | Le modèle entraîné, chargé par l'API au démarrage. Présenté comme « V2 » au livrable du 12/11 ; le fichier garde le nom `v1`. |
| `report_v1.md` | Scores en validation croisée (82 %, AUC 0,88) et variables les plus utiles. |
| `validation_30.md` | Résultat sur les 30 cas de référence (73 %, AUC 0,77 ; échantillon trop petit, voir le fichier). |

## `outputs/` : les scans

Un fichier par scan, nommé d'après le dernier mois complet utilisé.

| Fichier | Rôle |
|---|---|
| `scan_202610.csv` | Scan « fin octobre 2026 » : 95 décollages entre 2025-10 et 2026-06, dont 49 valides. |
| `scan_202312.csv` | Scan de contrôle « fin 2023 » : retrouve 37 cas d'entraînement avec les mêmes chiffres. |

## Comment marche le scan

Pour chaque page :

1. On récupère ses vues mensuelles **depuis 2018** (l'historique sert à savoir ce qui est normal pour la page).
2. On cherche un décollage **seulement dans la fenêtre** : t0 entre `dernier mois - 12` et `dernier mois - 4` (12 derniers mois, avec au moins 4 mois de recul).
3. Décollage = premier mois où les vues ≥ 1,5 × le record des 24 mois d'avant **et** la moyenne 3 mois ≥ 1,2 × ce record.
4. Pas de décollage → page ignorée. Décollage → garde-fous (trop petit, saisonnier), calcul des 9 variables avec les mois ≤ t0 + 4, prédiction.

L'entraînement utilise exactement la même règle, avec des décollages entre 2020 et 2023 dont on connaît déjà la fin.

## Règle d'or

`train.py`, `scan.py` et `api.py` importent `features.py`. **On ne recode jamais les variables ailleurs** : sinon les prédictions deviennent fausses sans erreur visible.

## Pour aller plus loin

- `README.md` : installation et commandes.
- `docs/API.md` : mode d'emploi de l'API pour l'équipe tech.
- `CLAUDE.md` : contexte complet du projet (définitions, choix, limites).
