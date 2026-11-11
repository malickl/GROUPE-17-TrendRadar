# Givaudan AI Challenge 2026 – Thème 3 : buzz ou tendance durable (V1)

## Contexte (à lire en premier)
- Problème : Givaudan (arômes et parfums) ne sait pas distinguer un **buzz** d'une **tendance durable**.
- Utilisateur cible : le manager CMI (Consumer Market Insights). Il doit décider, pour chaque sujet émergent et par région, s'il faut le suivre ou l'ignorer, avec des arguments.
- Objectif V1 : **4 mois après le décollage d'un sujet (t0), prédire s'il sera encore là 2 ans plus tard (tendance durable) ou retombé (buzz).** Un LLM expliquera ensuite pourquoi, en citant les données et les sources (étape ultérieure).
- Source du modèle : **Wikipédia seul**, avec les vues mensuelles de la page anglaise. Google Trends et Open Food Facts ont été testés : ils n'améliorent pas la prédiction. Ils serviront plus tard à l'explication.
- Échéances : l'API du modèle doit fonctionner le **11/11/2026**, la V1 complète le **vendredi 13/11/2026**.
- Style de l'utilisateur (Malick, étudiant data/IA, en français) :
  - réponses synthétiques et claires, sans jargon ;
  - demander avant de supposer le sens d'une colonne ;
  - prévenir avant de lancer quelque chose de long.

## Fichiers
| Fichier | Contenu |
|---|---|
| `data/train_v1_t0_plus_4mois.csv` | 175 cas étiquetés (98 Buzz, 77 Tendance durable). Colonnes : `title`, `group`, `t0` (AAAAMM), 8 colonnes numériques, `label`. |
| `data/wiki_series_2018_2026.tsv` | Vues mensuelles de 2 078 pages, avec `title`, `category`, et les colonnes `0`..`105` = 2018-01 → 2026-10. Sert aux tests hors ligne et de liste de pages à scanner. |
| `data/exclusions.txt` | 97 titres hors sujet ou anomalies, à ignorer (ex. Dragon fruit : page fusionnée). |
| `src/features.py` | **Calcul commun** : moyenne 3 mois, règle de décollage (t0), colonnes, garde-fous. `python src/features.py` vérifie qu'il reproduit les 175 cas à l'identique (attendu : `175/175`). |

**Règle d'or :** l'entraînement, le scan et l'API importent `src/features.py`. On ne recode jamais les colonnes ailleurs, sinon les prédictions deviennent fausses sans erreur visible.

## Définitions (résumé, le détail est dans `features.py`)
- **Courbe :** vues mensuelles d'une page Wikipédia anglaise, avec `agent=user` (les robots sont exclus).
- **Moyenne glissante :** sur 3 mois, non centrée (le mois courant et les 2 précédents), pour ne jamais utiliser le futur.
- **Record :** la plus haute moyenne glissante des 24 mois avant le mois testé, avec un plancher de 50.
- **t0 (décollage) :** premier mois où les deux conditions sont vraies :
  - vues brutes ≥ 1,5 × record ;
  - moyenne 3 mois ≥ 1,2 × record.
- **Colonnes du modèle :** `group`, `base`, `record_24m`, `ratio_now`, `max_ratio`, `jump_t0`, `growth_since_t0`, `volatility`, `seasonality`. Elles sont calculées avec les mois ≤ t0+4 uniquement.
- **Étiquette (entraînement seulement) :**
  - persistance = (moyenne des 12 derniers mois / pic sur [t0, t0+12]) / facteur de déclin de la catégorie ;
  - ≥ 0,7 = Tendance durable, < 0,4 = Buzz, entre les deux = exclu.
- **`group` :** 7 valeurs : Desserts, Boissons, Arômes & sucre, Fruits tropicaux, Olfactif, Épices, Noix & graines. La correspondance avec les catégories Wikipédia est dans `GRP`.
- **Variables les plus utiles :** `volatility` (un buzz fait des montagnes russes), `growth_since_t0` (continue-t-il de monter ?), `max_ratio` (hauteur du pic).

## Résultats de référence
- Hasard : 56 % (on répond toujours Buzz).
- Orange, Leave one out, k = 4 : Gradient Boosting 81 % (AUC 0,88), Random Forest 80 % (AUC 0,87).
- scikit-learn, 10 plis × 5 répétitions : Random Forest 82 % (AUC 0,89).
- Marge d'erreur avec 175 cas : environ ±6 points.

---

## Étape 1 – Entraîner et exporter le modèle (`src/train.py`)
1. Lire `data/train_v1_t0_plus_4mois.csv`. X = `FEATURES` (voir `features.py`), y = `label == 'Tendance durable'`.
2. Pipeline scikit-learn :
   - `ColumnTransformer` avec `OneHotEncoder(handle_unknown='ignore')` sur `group` et passthrough pour le reste ;
   - puis `RandomForestClassifier(n_estimators=500, min_samples_leaf=2, random_state=0)`.
3. Valider avec `RepeatedStratifiedKFold(10, 5)` et afficher précision et AUC. Attendu : environ 0,80 / 0,88. Comparer avec `HistGradientBoostingClassifier` et garder le meilleur.
4. Entraîner sur les 175 cas, puis sauvegarder `models/model_v1.joblib` avec :
   - `{pipeline, features, k=4, trained_on, cv_scores, version}` ;
   - les importances des variables, pour l'explication.
5. Sortie : un petit rapport (scores et top variables) dans `models/report_v1.md`.

## Étape 2 – Scanner Wikipédia pour les sujets actuels (`src/scan.py`)
**But :** trouver les sujets qui décollent en 2026 et que l'on peut déjà juger, c'est-à-dire avec t0 + 4 ≤ dernier mois complet.

1. **Pages à scanner :**
   - les 2 078 titres de `data/wiki_series_2018_2026.tsv`, moins `exclusions.txt` ;
   - option : rafraîchir la liste via les catégories Wikipédia (`Category:Desserts`, `Category:Non-alcoholic drinks`, `Category:Flavors`, `Category:Sugar substitutes`, `Category:Spices`, `Category:Edible nuts and seeds`, `Category:Tropical fruit`, `Category:Perfume ingredients`, `Category:Essential oils`). Utiliser l'API `action=query&list=categorymembers&cmlimit=500`, avec un niveau de sous-catégories maximum.
2. **Télécharger les vues mensuelles** depuis 2018-01 :
   ```
   https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user/{TITRE_URLENCODÉ}/monthly/2018010100/{AAAAMM01}00
   ```
   - **En-tête `User-Agent` obligatoire**, sinon 403 : `TrendRadarBot/0.1 (contact: mahassadiadou@gmail.com)`. Sans contact, Wikimedia limite fortement (erreurs 429).
   - **Confidentialité :** ne jamais mettre le nom « Givaudan » dans ce qui part sur le web (User-Agent, requêtes, URL, paramètres), pour Wikipédia comme pour Google Trends, Open Food Facts, GDELT, etc.
   - Les espaces deviennent `_` et le titre est encodé avec `quote(..., safe='')`.
   - En cas de 429 : backoff exponentiel, et 5 à 10 requêtes en parallèle au maximum.
   - Les mois absents valent 0. Ne garder que les mois **complets** : exclure le mois en cours.
   - Mettre en cache sur disque (`cache/pageviews/{titre}.json`).
3. Soit `last` le dernier mois complet. Chercher t0 entre `last-12` et `last-4` : décollage dans les 12 derniers mois, avec au moins 4 mois de recul (`window()` dans `scan.py` ; option `--since AAAAMM` pour changer le début). Ex. fin octobre 2026 → t0 de 2025-10 à 2026-06. (Malick a testé « depuis janvier de l'année » et préfère les 12 mois glissants.)
4. Appliquer `exclusion_reason(v, t0)`, puis noter la raison ou garder le cas.
5. Calculer `compute_features(v, t0)`, puis `group = GRP[category]`.
6. Prédire avec le modèle de l'étape 1, en sortant la probabilité de tendance.
7. Sortie : `outputs/scan_AAAAMM.csv` avec `title`, `group`, `t0`, les colonnes, `proba_tendance`, `label`, `verdict_verifiable_le` (t0 + 24 mois) et `exclu_raison`. Trier par probabilité.
8. **Contrôle :** relancer le scan en se mettant fin 2023 (`last` = 2023-12). Il doit retrouver des cas d'entraînement avec les mêmes colonnes.

## Étape 3 – API (`src/api.py`, FastAPI + uvicorn)
Elle charge `models/model_v1.joblib` au démarrage et réutilise `features.py` et le code de téléchargement de l'étape 2.

- `GET /health` → `{status, model_version, k}`.
- `POST /predict` avec `{"title": "Dubai chocolate", "category": "Desserts"}` (`group` optionnel).
  - Elle télécharge la courbe, détecte t0 dans les 12 derniers mois jouables, calcule les colonnes et prédit.
  - Réponse : `{title, t0, label, proba_tendance, features, top_raisons, verdict_verifiable_le, source: "Wikipedia pageviews (en, user)", series}`.
  - `top_raisons` reprend les 3 variables clés en phrases simples, par exemple « l'intérêt a encore augmenté de x2,1 depuis le décollage ».
  - S'il n'y a pas de t0, renvoyer `200` avec `label: null` et un message clair, du type « pas de décollage détecté ».
  - Si la page est introuvable, renvoyer `404`.
- `POST /predict_features` avec les 9 colonnes déjà calculées → prédiction. Utile pour tester et pour le futur LLM.
- `GET /scan/latest` → contenu du dernier `outputs/scan_*.csv`.
- `POST /scan/run` (`last`, `since` optionnels) → lance un scan en arrière-plan (202 ; 409 si déjà en cours). `GET /scan/status` → avancement, résultat ou erreur.
- Scan automatique : toutes les `SCAN_AUTO_HOURS` heures (défaut 24, 0 = désactivé), l'API lance le scan s'il manque celui du dernier mois complet. Choix de Malick : c'est l'équipe tech qui règle la fréquence.
- Si plus de 5 % des pages échouent au téléchargement, le scan n'écrit rien (on garde l'ancien fichier).
- Ajouter des tests avec `pytest` :
  - `/predict_features` sur 3 lignes du CSV d'entraînement donne la même prédiction que le modèle en direct ;
  - `/health` répond.
- Lancement : `uvicorn src.api:app --reload`, documentation sur `http://127.0.0.1:8000/docs`.

## Étape 4 – Plus tard (après l'API)
- Explication par LLM. Le LLM reçoit :
  - les colonnes et la courbe ;
  - Google Trends (intérêt 0-100 mensuel, via trends.google.com, en cherchant par « sujet » Freebase quand il existe) ;
  - Open Food Facts (produits lancés, via `search.openfoodfacts.org/search?q=product_name.en:"..."`, une requête par champ car « OR » renvoie une erreur 500 ; exclure les mois d'import en masse 2017-03 et 2020-04) ;
  - GDELT (nombre d'articles ; très limité, une requête toutes les 5 secondes, souvent bloqué).
  
  Il produit un argumentaire : pourquoi buzz ou tendance, et ce qui est pertinent pour Givaudan (ingrédient, arôme ou odeur).
- Version par pays : versions de Wikipédia en fr, de, es, ar, ja, pt via `langlinks`, et Google Trends par pays.
- **Élargir la liste des pages surveillées** (voulu par Malick après la V1) :
  - rafraîchir la liste avant chaque scan via `list=categorymembers` sur les 9 catégories, pour capter les nouvelles pages ;
  - puis élargir : sous-catégories plus profondes (ex. Desserts → Cakes → Chocolate cakes) et catégories voisines (thés, herbes aromatiques, fruits non tropicaux, cocktails…).
  - À prévoir : rattacher chaque nouvelle catégorie à un `group` (dans `GRP`), filtrer le hors-sujet (restaurants, marques), et garder en tête que le scan s'allonge (2 requêtes par page) et que le modèle est moins fiable hors des 9 catégories d'entraînement.

## Limites à garder en tête
- Wikipédia mesure la **curiosité**, pas la consommation (exemple : l'avocat sort « buzz »). Les étiquettes sont vraies « selon notre définition, d'après Wikipédia ».
- Il n'y a que 175 cas, dont 101 desserts. Les noix et les épices sont peu représentées.
- Une page renommée ou fusionnée peut créer un faux décollage : vérifier à la main les cas avec un saut énorme (`jump_t0` très élevé, `base` proche de 0).
- **Liste de pages figée (choix V1) :** le scan surveille toujours les mêmes 1 981 pages, issues de la photo `wiki_series_2018_2026.tsv` (9 catégories, au plus un niveau de sous-catégories), sur environ 7 millions d'articles du Wikipédia anglais. Une page créée après cette photo, ou rangée plus profondément, est invisible pour le radar. On garde ce fonctionnement pour la V1 du 13/11 ; l'élargissement est prévu à l'étape 4.
