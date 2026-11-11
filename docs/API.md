# API Radar de tendances : buzz ou tendance durable

Cette API détecte les sujets alimentaires et olfactifs (desserts, boissons, épices, parfums…) qui **décollent** sur Wikipédia. Pour chacun, elle prédit s'il s'agit d'un **buzz** (il retombera) ou d'une **tendance durable** (il sera encore là dans 2 ans).

---

## 1. Les notions à connaître

| Terme | Signification |
|---|---|
| **Signal / sujet** | Une page Wikipédia anglaise, par exemple `Maritozzo` ou `Açai bowl`. |
| **Décollage (`t0`)** | Premier mois où les vues de la page dépassent nettement son record des 2 ans précédents : au moins 1,5 fois ce record. |
| **Fenêtre analysée** | Seuls les décollages des **12 derniers mois** avec au moins **4 mois de recul** sont jugés. Fin octobre 2026, cela correspond aux décollages d'octobre 2025 à juin 2026. |
| **Tendance durable / Buzz** | La prédiction : l'intérêt sera-t-il encore là 2 ans après le décollage ? |
| **`proba_tendance`** | Probabilité de tendance, de 0 à 1. À partir de 0,5, la prédiction est « Tendance durable » ; en dessous, « Buzz ». |
| **Mois** | Toujours une **chaîne** au format `AAAAMM`, par exemple `"202601"` pour janvier 2026. |

Le modèle a 82 % de bonnes réponses : il se trompe environ une fois sur cinq. Il mesure la **curiosité** sur Wikipédia, pas les ventes.

---

## 2. Lancer l'API

**Prérequis :** Python récent (testé avec Python 3.15).

```bash
pip install -r requirements.txt
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

Une fois l'API lancée :
- **documentation interactive** (tester chaque appel depuis le navigateur) : `http://localhost:8000/docs` ;
- **adresse de base des appels** : `http://localhost:8000`.

### Variables d'environnement (optionnelles)

| Variable | Défaut | Rôle |
|---|---|---|
| `SCAN_AUTO_HOURS` | `24` | Toutes les N heures, l'API vérifie que le scan du dernier mois complet existe, et le lance s'il manque. `0` désactive cette vérification : vous déclenchez alors les scans vous-mêmes avec `POST /scan/run`. |
| `CORS_ORIGINS` | `*` | Origines autorisées à appeler l'API depuis un navigateur, séparées par des virgules. Exemple : `https://mon-front.app,http://localhost:3000`. |
| `DEMO_MODE` | `1` | `1` : `/scan/latest` sert le fichier de démo enrichi de champs fictifs (voir « Mode démo » en section 4). `0` : il sert le vrai dernier scan, sans champs fictifs. |

Exemple : `SCAN_AUTO_HOURS=6 CORS_ORIGINS=http://localhost:3000 uvicorn src.api:app --port 8000`

### Si l'API est partagée via ngrok

- L'adresse de base devient `https://xxxx.ngrok-free.app`.
- Ajoutez l'en-tête **`ngrok-skip-browser-warning: true`** à chaque requête. Sans lui, ngrok renvoie une page HTML d'avertissement au lieu du JSON.
- Si un mot de passe est configuré côté ngrok, ajoutez aussi `Authorization: Basic <base64("utilisateur:motdepasse")>`.

---

## 3. Les appels en un coup d'œil

| Méthode | Route | Entrée | Sortie | Usage dans l'interface |
|---|---|---|---|---|
| `GET` | `/health` | aucune | état de l'API | vérifier que le serveur répond |
| `GET` | `/scan/latest` | aucune | **la liste de tous les signaux récents avec leur prédiction** | **écran principal** (liste, carte, tableau de bord) |
| `POST` | `/predict` | `{title, category}` | prédiction détaillée d'un sujet + courbe | fiche détail ou recherche d'un sujet |
| `POST` | `/predict_features` | les 9 variables | prédiction | tests, usages avancés |
| `POST` | `/scan/run` | aucune (paramètres optionnels) | « lancé » | forcer la mise à jour de la liste |
| `GET` | `/scan/status` | aucune | avancement du scan | barre de progression, supervision |

**Comment ça marche :**
- `/scan/latest` **ne calcule rien** : il renvoie un tableau calculé à l'avance par le **scan**. Le scan passe en revue environ 2 000 pages Wikipédia, trouve celles qui décollent et prédit pour chacune. La réponse est donc instantanée.
- `/predict` calcule **en direct** pour un seul sujet : il télécharge la courbe Wikipédia, puis prédit. Comptez environ 1 seconde.

---

## 4. Détail de chaque appel

### `GET /health`

Indique si l'API fonctionne.

**Réponse `200`**
```json
{
  "status": "ok",
  "model_version": "v1",
  "k": 4,
  "model": "Random Forest"
}
```
- `k` : nombre de mois de recul après le décollage avant de prédire.

---

### `GET /scan/latest`  ⭐ appel principal

Renvoie tous les signaux du dernier scan. Ils sont déjà triés : d'abord les cas valides, par probabilité de tendance décroissante, puis les cas exclus.

**Réponse `200`**
```json
{
  "file": "scan_202610.csv",
  "last_month": "202610",
  "n": 95,
  "rows": [
    {
      "title": "Maritozzo",
      "group": "Desserts",
      "t0": "202601",
      "base": 114,
      "record_24m": 189,
      "ratio_now": 22.69,
      "max_ratio": 22.69,
      "jump_t0": 5.77,
      "growth_since_t0": 6.87,
      "volatility": 0.28,
      "seasonality": -0.31,
      "proba_tendance": 0.96,
      "label": "Tendance durable",
      "verdict_verifiable_le": "202801",
      "exclu_raison": null
    }
  ]
}
```

| Champ | Signification |
|---|---|
| `file` | Fichier source du scan. |
| `last_month` | Dernier mois complet utilisé. Les données vont jusqu'à fin octobre 2026 pour `202610`. |
| `n` | Nombre de lignes dans `rows`. |
| `rows[].title` | Nom de la page Wikipédia anglaise. Lien : `https://en.wikipedia.org/wiki/` + titre, en remplaçant les espaces par `_`. |
| `rows[].group` | Famille du sujet (voir les valeurs possibles en section 5). |
| `rows[].t0` | Mois du décollage. |
| `rows[].proba_tendance` | Probabilité de tendance durable, de 0 à 1. |
| `rows[].label` | `"Tendance durable"` ou `"Buzz"`. |
| `rows[].verdict_verifiable_le` | Date à laquelle on saura si la prédiction était juste (décollage + 24 mois). Informatif. |
| `rows[].exclu_raison` | `null` si le cas est **valide**. Sinon, la raison de l'exclusion (section 5). **Conseil : n'afficher par défaut que les lignes où ce champ vaut `null`.** |
| `rows[].base` … `seasonality` | Les variables utilisées par le modèle (section 5). |

**Erreur `404`** : aucun scan n'a encore été produit.

#### Mode démo (actif par défaut)

Pour la démo, `/scan/latest` sert un fichier **fixe**, `data/demo_scan.json` : le vrai scan d'octobre 2026, auquel on ajoute des **champs fictifs**. La réponse est identique à chaque appel.

La démo contient **45 sujets** :
- uniquement les cas **valides** : `exclu_raison` vaut toujours `null` ;
- sans 4 sujets hors-sujet ou trop génériques : Mast seeding, Types of cocoa beans, Fragrance compound et Crazy Mountain.

- **Ne changent pas** (réels, issus du modèle) : `title`, `group`, `t0`, `label`, `proba_tendance`, les 8 variables, `verdict_verifiable_le` et `exclu_raison`.
- **Ajoutés au niveau global :** `demo_fictif: true`, `avertissement` (texte qui liste les champs fictifs) et `source_reelle`.
- **Ajoutés à chaque sujet (fictifs) :**

| Champ | Type | Contenu |
|---|---|---|
| `demo_fictif` | booléen | Toujours `true`. |
| `domaine` | texte | `"Parfum"` si `group` vaut `Olfactif`, sinon `"Goût"`. |
| `country` | texte | Code ISO du **pays d'origine** du sujet quand il est connu (ex. Maritozzo → `IT`, Halawet el jibn → `LB`, Thekua → `IN`). Sinon, tiré au hasard parmi FR, US, GB, DE, IT, ES, BR, MX, IN, JP. |
| `country_source` | texte | `"origine"` (pays d'origine du sujet) ou `"aleatoire"` (tiré au hasard). |
| `lat`, `lon` | nombres | Coordonnées de la capitale du pays, avec un léger décalage (±0,6°) pour que les points ne se superposent pas sur une carte. |
| `element_cle` | texte | Élément gustatif ou olfactif qui fait le succès du sujet, en une phrase. |
| `veille.resume` | texte | Ce qui se dirait sur internet à propos du sujet, en 2-3 phrases. |
| `veille.sources` | liste de `{titre, url}` | 3 liens **réels** : la page Wikipédia du sujet, sa recherche Google Trends et sa recherche Google Actualités. |
| `matieres_premieres` | liste de textes | 2 à 4 matières premières liées au sujet. |
| `date_prediction` | `AAAAMM` | Mois de la prédiction, entre janvier et octobre 2026, postérieur à `t0`. |
| `historique` | liste de `{mois, label, proba_tendance}` | Une entrée par mois, de `date_prediction` à `202610`. Le verdict reste le même sur toute la période, et le dernier mois porte la vraie `proba_tendance`. |

Exemple (extrait) :
```json
{
  "title": "Maritozzo", "group": "Desserts", "t0": "202601",
  "label": "Tendance durable", "proba_tendance": 0.96,
  "domaine": "Goût", "country": "US", "lat": 38.9102, "lon": -77.0227,
  "element_cle": "La brioche moelleuse fendue et débordante de crème fouettée vanillée",
  "veille": {
    "resume": "La brioche romaine à la crème s'affiche dans les vitrines des boulangeries hors d'Italie et sur les réseaux sociaux, où la coupe généreuse de crème fait le visuel. Les pâtissiers la déclinent en versions pistache, chocolat ou agrumes.",
    "sources": [
      { "titre": "Wikipédia : Maritozzo", "url": "https://en.wikipedia.org/wiki/Maritozzo" },
      { "titre": "Google Trends : intérêt de recherche pour « Maritozzo »", "url": "https://trends.google.com/trends/explore?date=today%2012-m&q=Maritozzo" },
      { "titre": "Google Actualités : articles sur « Maritozzo »", "url": "https://news.google.com/search?q=Maritozzo" }
    ]
  },
  "matieres_premieres": ["Farine de blé", "Crème fraîche", "Vanille", "Zeste d'orange"],
  "date_prediction": "202607",
  "historique": [
    { "mois": "202607", "label": "Tendance durable", "proba_tendance": 0.966 },
    { "mois": "202608", "label": "Tendance durable", "proba_tendance": 0.963 },
    { "mois": "202609", "label": "Tendance durable", "proba_tendance": 0.976 },
    { "mois": "202610", "label": "Tendance durable", "proba_tendance": 0.96 }
  ],
  "demo_fictif": true
}
```

Le fichier se régénère avec `python src/demo.py`. La graine aléatoire est fixe, donc le résultat est toujours le même. Les textes sont rédigés dans `data/demo_contenu.json`.

---

### `POST /predict`

Prédiction pour un sujet précis. L'API télécharge la courbe Wikipédia, cherche un décollage dans la fenêtre, puis prédit.

**Requête**
```json
{ "title": "Maritozzo", "category": "Desserts" }
```
| Champ | Obligatoire | Valeurs |
|---|---|---|
| `title` | oui | Titre **exact** de la page Wikipédia anglaise. |
| `category` | oui, sauf si `group` est fourni | Catégorie Wikipédia (section 5). |
| `group` | non (prioritaire sur `category`) | Famille du sujet (section 5). |

Paramètres d'URL optionnels :
- `?last=AAAAMM` : dernier mois à utiliser (défaut : le dernier mois complet) ;
- `?since=AAAAMM` : début de la fenêtre de décollage (défaut : 12 mois avant `last`).

**Réponse `200` : décollage trouvé**
```json
{
  "title": "Maritozzo",
  "group": "Desserts",
  "source": "Wikipedia pageviews (en, user)",
  "last_month": "202610",
  "series": {
    "months": ["201801", "201802", "201803", "..."],
    "views":  [0, 0, 0, "..."]
  },
  "t0": "202601",
  "label": "Tendance durable",
  "proba_tendance": 0.96,
  "features": {
    "group": "Desserts", "base": 114, "record_24m": 189, "ratio_now": 22.69,
    "max_ratio": 22.69, "jump_t0": 5.77, "growth_since_t0": 6.87,
    "volatility": 0.28, "seasonality": -0.31
  },
  "top_raisons": [
    {
      "variable": "growth_since_t0",
      "valeur": 6.87,
      "phrase": "l'intérêt a encore augmenté de x6,9 depuis le décollage : plutôt signe de tendance",
      "mediane_buzz": 0.57,
      "mediane_tendance": 0.95
    },
    {
      "variable": "jump_t0",
      "valeur": 5.77,
      "phrase": "au décollage, l'intérêt a été multiplié par 5,8 en un mois : plutôt signe de buzz",
      "mediane_buzz": 1.98,
      "mediane_tendance": 1.39
    },
    {
      "variable": "volatility",
      "valeur": 0.28,
      "phrase": "la courbe est régulière depuis le décollage (volatilité 0,28) : plutôt signe de tendance",
      "mediane_buzz": 0.65,
      "mediane_tendance": 0.28
    }
  ],
  "verdict_verifiable_le": "202801"
}
```

| Champ | Signification |
|---|---|
| `label` | `"Tendance durable"` ou `"Buzz"`. Même nom dans `/scan/latest` et `/predict_features`. |
| `proba_tendance` | Probabilité de tendance, de 0 à 1. |
| `t0` | Mois du décollage. |
| `series` | **Courbe complète** des vues mensuelles depuis janvier 2018 : `months[i]` correspond à `views[i]`. À afficher en graphique, idéalement avec un marqueur sur `t0`. |
| `features` | Les 9 variables calculées (section 5). |
| `top_raisons` | Les 3 arguments principaux, en phrases prêtes à afficher. `mediane_buzz` et `mediane_tendance` donnent la valeur typique d'un buzz et d'une tendance, par exemple pour une jauge. |
| `verdict_verifiable_le` | Date à laquelle on saura si la prédiction était juste. |
| `avertissement` | *Présent seulement si* le cas aurait été écarté par le scan (trop petit ou saisonnier) : la prédiction est alors moins fiable. |

**Réponse `200` : pas de décollage dans la fenêtre** (ce n'est pas une erreur)
```json
{
  "title": "Dubai chocolate",
  "group": "Desserts",
  "source": "Wikipedia pageviews (en, user)",
  "last_month": "202610",
  "series": { "months": ["201801", "..."], "views": [0, "..."] },
  "t0": null,
  "label": null,
  "proba_tendance": null,
  "message": "Pas de décollage détecté entre 202510 et 202606 : il faut un mois avec au moins 1,5x le record des 2 ans d'avant, et 4 mois de recul."
}
```
→ Tester `label === null` et afficher `message`. La courbe `series` reste disponible.

**Erreurs**
| Code | Cas | Exemple de `detail` |
|---|---|---|
| `404` | Page Wikipédia inexistante (vérifier l'orthographe exacte du titre) | `"Page Wikipédia anglaise introuvable : « Zzzz »"` |
| `422` | Ni `category` ni `group` valide | `"Indiquer 'category' (...) ou 'group' (...)."` |
| `503` | Wikipédia ne répond pas | `"Wikipédia ne répond pas : ..."` |

---

### `POST /predict_features`

Prédiction à partir des 9 variables déjà calculées, sans téléchargement. Utile pour les tests.

**Requête**
```json
{
  "group": "Desserts", "base": 114, "record_24m": 189, "ratio_now": 22.69,
  "max_ratio": 22.69, "jump_t0": 5.77, "growth_since_t0": 6.87,
  "volatility": 0.28, "seasonality": -0.31
}
```
Les 9 champs sont obligatoires.

**Réponse `200`**
```json
{
  "label": "Tendance durable",
  "proba_tendance": 0.96,
  "features": { "group": "Desserts", "base": 114.0, "...": "..." },
  "top_raisons": [ { "variable": "growth_since_t0", "valeur": 6.87, "phrase": "...", "mediane_buzz": 0.57, "mediane_tendance": 0.95 } ]
}
```
**Erreur `422`** : `group` inconnu ou champ manquant.

---

### `POST /scan/run`

Lance un **nouveau scan en arrière-plan**. La réponse arrive tout de suite ; le calcul prend quelques minutes, le temps de télécharger environ 2 000 pages Wikipédia. Pendant ce temps, `/scan/latest` continue de renvoyer l'ancien tableau, puis bascule automatiquement sur le nouveau.

Paramètres d'URL optionnels : `?last=AAAAMM` (défaut : dernier mois complet) et `?since=AAAAMM` (défaut : 12 mois avant).

**Réponse `202`**
```json
{ "status": "lancé", "last_month": "202610", "suivi": "/scan/status" }
```

**Erreurs**
| Code | Cas |
|---|---|
| `409` | Un scan est déjà en cours. |
| `422` | `last` ou `since` mal formé (attendu `AAAAMM`), ou `last` dans le futur. |

Le scan est conçu pour tourner **une fois par mois**, au début du mois. Le relancer plus souvent ne change rien : les données Wikipédia sont mensuelles.

---

### `GET /scan/status`

État du scan en cours, ou du dernier scan lancé par l'API.

**Réponse `200` : scan en cours**
```json
{
  "running": true,
  "trigger": "manuel",
  "last_month": "202610",
  "since": null,
  "started_at": "2026-11-10T15:02:11",
  "finished_at": null,
  "progress": "850/1981 pages",
  "result": null,
  "error": null,
  "auto": { "actif": true, "verification_toutes_les_heures": 24.0 },
  "dernier_fichier": "scan_202610.csv",
  "dernier_mois_complet": "202610"
}
```

**Réponse `200` : scan terminé**
```json
{
  "running": false,
  "trigger": "manuel",
  "last_month": "202610",
  "finished_at": "2026-11-10T15:06:40",
  "progress": "1981/1981 pages",
  "result": {
    "file": "scan_202610.csv",
    "decollages": 95,
    "valides": 49,
    "tendance": 31,
    "buzz": 18,
    "pages_en_erreur": 0
  },
  "error": null,
  "...": "..."
}
```

| Champ | Signification |
|---|---|
| `running` | `true` pendant un scan. |
| `trigger` | `"manuel"` (via `/scan/run`) ou `"auto"` (vérification automatique). |
| `progress` | Avancement du téléchargement. |
| `result` | Résumé du scan terminé. |
| `error` | Message si le scan a échoué. Dans ce cas, **l'ancien tableau est conservé**. Par exemple, si Wikipédia bloque plus de 5 % des pages, rien n'est écrasé. |
| `dernier_fichier` | Fichier actuellement servi par `/scan/latest`. |

---

## 5. Référence des valeurs

### `group` (7 familles)
`Desserts`, `Boissons`, `Arômes & sucre`, `Fruits tropicaux`, `Olfactif`, `Épices`, `Noix & graines`

### `category` (catégories Wikipédia) → `group`
| `category` | `group` |
|---|---|
| `Desserts` | Desserts |
| `Non-alcoholic drinks` | Boissons |
| `Flavors`, `Sugar substitutes` | Arômes & sucre |
| `Tropical fruit` | Fruits tropicaux |
| `Perfume ingredients`, `Essential oils` | Olfactif |
| `Spices` | Épices |
| `Edible nuts and seeds` | Noix & graines |

### `exclu_raison`
| Valeur | Signification |
|---|---|
| `null` | Cas valide. |
| `"trop petit (<500 vues/mois en moyenne 3 mois)"` | Page trop peu consultée : ce « décollage » est du bruit. |
| `"saisonnier (même mois l'an passé déjà >=70%)"` | Pic qui revient chaque année (fêtes, saisons) : ce n'est pas un nouveau sujet. |

### Les 9 variables du modèle
| Variable | En clair |
|---|---|
| `group` | Famille du sujet. |
| `base` | Vues mensuelles habituelles avant le décollage (médiane sur 2 ans). |
| `record_24m` | Meilleur niveau (moyenne sur 3 mois) des 2 ans avant le décollage. |
| `ratio_now` | Niveau 4 mois après le décollage, divisé par l'ancien record. |
| `max_ratio` | Hauteur du pic, divisée par l'ancien record. |
| `jump_t0` | Brutalité du décollage : multiplication de l'intérêt en un mois. |
| `growth_since_t0` | Évolution depuis le décollage : au-dessus de 1, l'intérêt continue de monter. **Variable la plus importante.** |
| `volatility` | Irrégularité de la courbe depuis le décollage. Élevée = montagnes russes = plutôt un buzz. |
| `seasonality` | Ressemblance entre l'année -1 et l'année -2 (de -1 à 1). |

---

## 6. Exemples de code (JavaScript)

```js
const BASE = "http://localhost:8000";   // ou l'adresse ngrok
const HEADERS = { "ngrok-skip-browser-warning": "true" };  // inutile en local

// Liste des signaux valides
const scan = await fetch(`${BASE}/scan/latest`, { headers: HEADERS }).then(r => r.json());
const signaux = scan.rows.filter(s => s.exclu_raison === null);

// Fiche d'un sujet
const res = await fetch(`${BASE}/predict`, {
  method: "POST",
  headers: { ...HEADERS, "Content-Type": "application/json" },
  body: JSON.stringify({ title: "Maritozzo", category: "Desserts" }),
});
if (res.status === 404) { /* page introuvable */ }
const fiche = await res.json();
if (fiche.label === null) { /* afficher fiche.message */ }

// Forcer un nouveau scan, puis suivre l'avancement
await fetch(`${BASE}/scan/run`, { method: "POST", headers: HEADERS });
const status = await fetch(`${BASE}/scan/status`, { headers: HEADERS }).then(r => r.json());
```

En ligne de commande :
```bash
curl http://localhost:8000/scan/latest
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
     -d '{"title": "Maritozzo", "category": "Desserts"}'
curl -X POST http://localhost:8000/scan/run
curl http://localhost:8000/scan/status
```

---

## 7. Limites à connaître pour le design

- **Pas de dimension géographique réelle pour l'instant.** En mode démo, `country`, `lat` et `lon` sont fictifs. Toutes les données viennent du Wikipédia **anglais** : un signal est « mondial / anglophone ». Une vue par pays est prévue dans une version ultérieure. D'ici là, une carte ne peut pas afficher de différence par pays.
- **Liste de sujets fixe.** Le radar surveille une liste fixe de 1 981 pages Wikipédia, issues de 9 catégories (desserts, épices, arômes, boissons…). Un sujet absent de cette liste, par exemple une page créée récemment, n'apparaîtra pas dans `/scan/latest`. L'élargissement de la liste est prévu dans une version ultérieure.
- **Données mensuelles.** Le tableau change une fois par mois.
- **Titres exacts.** `/predict` attend le titre exact de la page Wikipédia anglaise. Une recherche approximative doit être faite côté interface, par exemple en proposant les titres de `/scan/latest`.
- **Une prédiction sur cinq est fausse** en moyenne. Affichez `proba_tendance` (par exemple « 96 % ») plutôt qu'un verdict seul.
- **Les explications détaillées** (sources, arguments rédigés) viendront dans une version ultérieure. Pour l'instant, utilisez `top_raisons`.
