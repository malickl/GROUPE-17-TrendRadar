# Radar de tendances : buzz ou tendance durable (V1)

Ce projet détecte les sujets food et parfum qui **décollent** sur Wikipédia. Quatre mois après le décollage, il prédit si le sujet sera encore là dans 2 ans (**tendance durable**) ou s'il sera retombé (**buzz**).

- Modèle : Random Forest entraîné sur 175 cas étiquetés, avec 82 % de bonnes réponses en validation croisée (AUC 0,89).
- Source : vues mensuelles du Wikipédia anglais.

## Démarrage rapide

Prérequis : Python récent (testé avec Python 3.15).

```bash
python -m venv .venv
source .venv/bin/activate          # Windows : .venv\Scripts\activate
pip install -r requirements.txt
uvicorn src.api:app --port 8000 --reload
```

Une fois l'API lancée :
- documentation interactive : http://127.0.0.1:8000/docs
- interface de démo (bulles) : http://127.0.0.1:8000/ui

**Documentation complète de l'API** (appels, réponses, champs, erreurs, exemples JS) : [docs/API.md](docs/API.md)

Les tests se lancent avec `pytest -q`.

## Structure

| Chemin | Contenu |
|---|---|
| `src/features.py` | Calcul commun des variables, utilisé partout. Ne jamais recoder ces variables ailleurs. |
| `src/train.py` | Entraînement du modèle. Produit `models/model_v1.joblib` et `models/report_v1.md`. |
| `src/scan.py` | Scan de Wikipédia. Produit `outputs/scan_AAAAMM.csv`. |
| `src/api.py` | L'API FastAPI. |
| `web/index.html` | L'interface de démo, servie sur `/ui`. |
| `tests/` | Les tests de l'API. |
| `data/` | Données d'entraînement, séries Wikipédia 2018-2026 et liste d'exclusions. |
| `models/` | Le modèle entraîné, livré pour ne pas avoir à le réentraîner. |
| `outputs/` | Les résultats de scan. `/scan/latest` sert le plus récent. |
| `CLAUDE.md` | Contexte complet du projet : définitions, choix et limites. |

## Commandes utiles

```bash
python src/features.py                              # vérifie que les variables sont reproduites (175/175)
python src/train.py                                 # réentraîne le modèle (~1 min)
python src/scan.py                                  # nouveau scan à partir de Wikipédia
python src/scan.py --asof 202312 --offline --check  # contrôle sur 2023, sans réseau
```
