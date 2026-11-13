# Livrable Data : reproduction, comparaison V1/V2, limites et biais

Ce document couvre les livrables Data du vendredi 13/11/2026 :
1. la procédure de reproduction des résultats ;
2. la comparaison V1/V2 avec métriques ;
3. l'analyse des limites et des biais.

Le code et la documentation sont décrits dans le [README](../README.md), l'[architecture](ARCHITECTURE.md) et le [mode d'emploi de l'API](API.md).

> **À propos des noms de version.** Le modèle présenté comme « V2 » (jeudi 12) est celui du code actuel. Ses fichiers gardent le nom `v1` (`models/model_v1.joblib`, `models/report_v1.md`) pour ne pas casser les chemins utilisés par l'API.

---

## 1. Procédure de reproduction

### Installation

```bash
python -m venv .venv
source .venv/bin/activate          # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

### Étapes et résultats attendus

Toutes les commandes se lancent depuis la racine du projet. Les étapes 1 à 5 n'ont **pas besoin d'internet** : elles utilisent les données livrées dans `data/`.

| # | Commande | Ce qu'elle vérifie | Résultat attendu |
|---|---|---|---|
| 1 | `python src/features.py` | Les 9 variables du modèle sont recalculées à l'identique à partir des courbes brutes. | `175/175 cas identiques` |
| 2 | `python src/train.py` | Réentraîne le modèle et le compare au Gradient Boosting (validation croisée 10 plis × 5). Durée : environ 1 min. | Random Forest **81,9 %**, AUC **0,889** (retenu) ; Gradient Boosting 78,3 %, AUC 0,849. Rapport : `models/report_v1.md`. |
| 3 | `python src/validate.py` | Prédit les 30 cas de référence sans les avoir vus à l'entraînement. | **22/30 (73,3 %)**, AUC **0,769**. Rapport : `models/validation_30.md`. |
| 4 | `python src/scan.py --asof 202312 --offline --check` | Rejoue le scan comme si on était fin 2023 et vérifie qu'il retrouve les cas d'entraînement. | `37 retrouvés, 37 identiques` |
| 5 | `pytest -q` | Tests de l'API (`/health`, `/predict_features` identique au modèle en direct, etc.). | `10 passed` |
| 6 | `python src/scan.py` | Scan réel sur Wikipédia (**internet nécessaire**, plusieurs minutes). | `outputs/scan_AAAAMM.csv`. Pour octobre 2026 : 45 sujets valides. |
| 7 | `uvicorn src.api:app --port 8000` | Lance l'API. | Documentation interactive sur http://127.0.0.1:8000/docs |

**Déterminisme :** les tirages aléatoires utilisent des graines fixes (`random_state=0` pour le modèle et la validation croisée, graine `2026` pour les 30 cas de référence). Les mêmes commandes donnent donc les mêmes chiffres.

### Chaîne complète

```
data/wiki_series_2018_2026.tsv ──► src/features.py ──► data/train_v1_t0_plus_4mois.csv (175 cas)
                                                              │
                                                              ▼
                                        src/train.py ──► models/model_v1.joblib
                                                              │
                     Wikipédia (API) ──► src/scan.py ─────────┤──► outputs/scan_AAAAMM.csv
                                                              ▼
                                                        src/api.py (FastAPI)
```

---

## 2. Comparaison V1 / V2

### Ce qui a changé

| | V1 (Gate 1, mardi 10) | V2 (prototype, jeudi 12) |
|---|---|---|
| Outil | Orange (interface graphique) | Code Python (scikit-learn), reproductible |
| Données | 175 cas Wikipédia, 9 variables | Les mêmes 175 cas et 9 variables, calcul centralisé dans `features.py` |
| Modèle | Gradient Boosting (meilleur dans Orange) | Random Forest, 500 arbres (meilleur en Python) |
| Validation | Leave one out | Validation croisée 10 plis × 5 répétitions, plus 30 cas de référence |
| Mise à disposition | Aucune | API (prédiction d'un sujet, scan mensuel automatique) |

### Métriques

| Version | Validation | Précision | AUC |
|---|---|---|---|
| Hasard (toujours « buzz ») | – | 56 % | 0,50 |
| V1 : Orange, Gradient Boosting | Leave one out (k = 4) | 81 % | 0,88 |
| V1 : Orange, Random Forest | Leave one out (k = 4) | 80 % | 0,87 |
| **V2 : Python, Random Forest** | **10 plis × 5 répétitions** | **82 %** (± 9) | **0,89** |
| V2 : Python, Gradient Boosting | 10 plis × 5 répétitions | 78 % | 0,85 |
| V2 sur les 30 cas de référence | Hors entraînement | 73 % | 0,77 |

**Lecture :**
- Les performances V1 et V2 sont **équivalentes**. L'écart (environ 1 point) est bien inférieur à la marge d'erreur, d'environ ± 6 points avec 175 cas.
- L'apport de la V2 n'est pas la précision : c'est un modèle **reproductible, testé et branché sur une API**.
- Le test sur 30 cas (73 %) est sous les seuils fixés à l'avance (75 % et AUC 0,80). Avec seulement 30 cas, la marge d'erreur est d'environ ± 15 points, donc la mesure fiable reste celle sur les 175 cas. Le tirage et les seuils n'ont **pas** été modifiés après coup.

### Variables qui comptent le plus (V2)

Mesure : baisse de l'AUC quand on mélange la colonne, sur des cas non vus.

| Variable | Signification | Importance |
|---|---|---|
| `growth_since_t0` | L'intérêt continue-t-il de monter après le décollage ? | 0,042 |
| `jump_t0` | Brutalité du saut au décollage | 0,036 |
| `volatility` | « Montagnes russes » de la courbe | 0,034 |
| `max_ratio` | Hauteur du pic par rapport à l'ancien record | 0,027 |

### Piste en cours (non livrée) : base d'entraînement élargie

Une expérience sur une branche séparée élargit la liste de pages, de 1 981 à 17 558 pages : thés, cafés, fromages, soins, régimes…

| | V2 livrée | Expérience « base élargie » |
|---|---|---|
| Cas d'entraînement | 175 | 1 599 |
| Validation croisée | 82 % (± 9) | 81 % (± 3) |
| Test sur les 175 cas de la V2 | 82 % | 80 % (AUC 0,89) |

Le résultat est **aussi bon, mais pas meilleur**. La mesure est en revanche plus fiable, avec une marge divisée par 3, et le modèle couvre de nouvelles familles. Cette piste n'est pas utilisée par l'API livrée.

---

## 3. Limites et biais

### Ce que mesurent les données
- **Wikipédia mesure la curiosité, pas la consommation.** Un produit très consommé mais peu recherché, comme l'avocat, peut sortir « buzz ». Les étiquettes sont vraies **selon notre définition, d'après Wikipédia**.
- **Wikipédia anglais seulement :** le modèle voit surtout l'intérêt des pays anglophones. Une tendance locale (Japon, Brésil…) peut passer inaperçue.
- **Baisse générale de Wikipédia :** les vues baissent depuis plusieurs années (assistants IA, réponses directes des moteurs). L'étiquette corrige cet effet par famille avec un « facteur de déclin », qui reste une approximation.

### Biais de la base d'entraînement
- **Petite base :** 175 cas, d'où une marge d'erreur d'environ ± 6 points.
- **Déséquilibre des familles :** 101 desserts sur 175. Les noix, les épices et le parfum sont peu représentés, donc le modèle y est moins fiable.
- **Définition choisie :** les seuils « tendance ≥ 70 % du pic, buzz < 40 % » sont un choix. Les cas entre les deux sont écartés, si bien que le modèle n'a jamais appris les cas ambigus.
- **Période 2020-2023 :** elle inclut la période Covid, avec des comportements de recherche inhabituels.

### Pièges techniques identifiés
- **Pages renommées, fusionnées ou redirigées :** elles créent de faux décollages ou des courbes trompeuses. Par exemple, la page « Ube » est une redirection : les lecteurs arrivent sur une autre page. Il faut donc le titre exact de la page. Les cas suspects sont exclus (`data/exclusions.txt`) ou signalés par des garde-fous.
- **Pages récentes :** une page créée pendant la période (par exemple Labubu, créée en octobre 2024) montre une « montée » qui n'est que sa création. L'expérience « base élargie » montre que ces cas sortent presque toujours « tendance ». Ils doivent être filtrés.
- **Liste de pages figée :** le scan surveille 1 981 pages issues de 9 catégories. Un sujet sans page, ou rangé ailleurs, est invisible pour le radar.
- **Délai :** le verdict arrive 4 mois après le décollage. C'est plus fiable qu'un verdict immédiat, mais ça laisse moins de temps pour agir.

### Limites de l'explication
- Les « raisons » renvoyées par l'API (`top_raisons`) comparent le sujet aux cas typiques sur les 3 variables les plus importantes **en général**. Ce n'est **pas** le raisonnement exact du modèle pour ce sujet, et elles peuvent se contredire. Une méthode comme SHAP donnerait les vraies contributions de chaque variable.

### Démo
- Dans le mode démo, certains champs sont **fictifs** : pays, carte, domaine, élément clé, matières premières, résumé de veille, historique. Ils sont marqués `demo_fictif: true` et ne modifient jamais le verdict du modèle.

### Prochaines étapes pour réduire ces limites
1. Ajouter des sources avec historique : Google Trends, presse (GDELT), Reddit, Wikipédia dans d'autres langues.
2. Élargir et nettoyer la base d'entraînement, en retirant les pages récentes et le hors-sujet.
3. Donner les vraies raisons de chaque verdict (SHAP), puis les faire expliquer par un LLM.
4. Garder chaque prédiction, la vérifier 2 ans plus tard et réentraîner le modèle.
