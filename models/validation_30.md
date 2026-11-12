# Validation sur 30 cas de référence

## Protocole

1. **Choix des cas** : 30 cas tirés au hasard (graine 2026) parmi les 175 cas étiquetés, 15 Tendance durable et 15 Buzz, répartis sur toutes les familles. Tirage fait avant de regarder les prédictions.
2. **Prédiction honnête** : chaque cas est prédit par un modèle qui ne l'a pas vu à l'entraînement (validation croisée 10 plis, même pipeline que le modèle publié).
3. **Comparaison** à la vraie étiquette (définie dans `features.py`).

Fichier des cas : `data/reference_30.csv`. Relancer : `python src/validate.py`.

## Critères de validation

| Critère | Seuil | Résultat | Validé |
|---|---|---|---|
| Battre le hasard (50 % sur 15/15) | > 50 % | 73% | ✅ |
| Précision | ≥ 75% | 73% | ❌ |
| AUC | ≥ 0.80 | 0.77 | ❌ |

> **À lire avec prudence : l'échantillon est trop petit.** Avec 30 cas, une erreur pèse plus de 3 points et la marge d'erreur est d'environ ±15 points. Ces 30 cas servent surtout à montrer concrètement, cas par cas, où le modèle a raison et où il se trompe. La mesure fiable est celle sur les 175 cas, avec la même méthode : précision 82%, AUC 0.88 (voir plus bas et `report_v1.md`). Les seuils ci-dessus ont été fixés avant de voir les résultats et n'ont pas été modifiés.

## Détail des erreurs

| | Prédit Tendance | Prédit Buzz |
|---|---|---|
| **Vraie Tendance** | 11 | 4 |
| **Vrai Buzz** | 4 | 11 |

## Pour comparaison : les 175 cas

Même méthode sur les 175 cas : précision 82%, AUC 0.88. Avec 30 cas, la marge d'erreur est d'environ ±15 points.

## Les 30 cas

| Sujet | Famille | t0 | Vraie réponse | Proba tendance | Prédiction | OK |
|---|---|---|---|---|---|---|
| Cyclamate | Arômes & sucre | 202201 | Tendance durable | 0.51 | Tendance durable | ✅ |
| Aspartame | Arômes & sucre | 202306 | Buzz | 0.17 | Buzz | ✅ |
| Synsepalum dulcificum | Arômes & sucre | 202304 | Buzz | 0.10 | Buzz | ✅ |
| Bunchosia glandulifera | Fruits tropicaux | 202005 | Tendance durable | 0.74 | Tendance durable | ✅ |
| Ensete ventricosum | Fruits tropicaux | 202201 | Buzz | 0.07 | Buzz | ✅ |
| Solanum pimpinellifolium | Fruits tropicaux | 202005 | Tendance durable | 0.70 | Tendance durable | ✅ |
| Basil seed | Noix & graines | 202305 | Tendance durable | 0.89 | Tendance durable | ✅ |
| Bazzini | Noix & graines | 202206 | Buzz | 0.53 | Tendance durable | ❌ |
| Brazil nut | Noix & graines | 202111 | Buzz | 0.05 | Buzz | ✅ |
| Lilial | Olfactif | 202201 | Buzz | 0.45 | Buzz | ✅ |
| Phenylacetic acid | Arômes & sucre | 202304 | Tendance durable | 0.57 | Tendance durable | ✅ |
| Alligator pepper | Épices | 202004 | Buzz | 0.80 | Tendance durable | ❌ |
| Shikanji | Boissons | 202304 | Buzz | 0.74 | Tendance durable | ❌ |
| Nigella sativa | Épices | 202003 | Buzz | 0.38 | Buzz | ✅ |
| Rosa × damascena | Épices | 202005 | Tendance durable | 0.43 | Buzz | ❌ |
| Uzazi | Épices | 202004 | Tendance durable | 0.13 | Buzz | ❌ |
| Perilla frutescens | Noix & graines | 202005 | Tendance durable | 0.54 | Tendance durable | ✅ |
| Crataegus mexicana | Fruits tropicaux | 202312 | Buzz | 0.02 | Buzz | ✅ |
| Solanum sessiliflorum | Fruits tropicaux | 202211 | Tendance durable | 0.67 | Tendance durable | ✅ |
| Twisted doughnut | Desserts | 202004 | Tendance durable | 0.61 | Tendance durable | ✅ |
| Sherbet (frozen dessert) | Desserts | 202108 | Tendance durable | 0.39 | Buzz | ❌ |
| London bun | Desserts | 202302 | Buzz | 0.23 | Buzz | ✅ |
| Ambrein | Olfactif | 202106 | Tendance durable | 0.91 | Tendance durable | ✅ |
| 3-Octanone | Olfactif | 202301 | Buzz | 0.15 | Buzz | ✅ |
| Safrole | Olfactif | 202309 | Tendance durable | 0.77 | Tendance durable | ✅ |
| Pine oil | Olfactif | 202003 | Buzz | 0.48 | Buzz | ✅ |
| Lamington | Desserts | 202308 | Buzz | 0.51 | Tendance durable | ❌ |
| Canned tea | Boissons | 202310 | Buzz | 0.49 | Buzz | ✅ |
| Champús | Boissons | 202108 | Tendance durable | 0.33 | Buzz | ❌ |
| Crodino | Boissons | 202208 | Tendance durable | 0.54 | Tendance durable | ✅ |
