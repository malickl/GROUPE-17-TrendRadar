# Rapport modèle v1

- Données : `train_v1_t0_plus_4mois.csv` : 175 cas (77 Tendance durable, 98 Buzz)
- Prédiction à t0 + 4 mois
- Hasard (toujours Buzz) : 56%
- Entraîné le 2026-11-10

## Validation croisée (10 plis x 5 répétitions)

| Modèle | Précision | AUC |
|---|---|---|
| Random Forest **(retenu)** | 81.9% ± 9.1% | 0.889 ± 0.079 |
| Gradient Boosting | 78.3% ± 9.6% | 0.849 ± 0.087 |

## Variables les plus utiles

Baisse d'AUC quand on mélange la colonne, mesurée sur des cas non vus (10 plis). Plus c'est haut, plus elle compte.

| Variable | Importance |
|---|---|
| `growth_since_t0` | 0.042 |
| `jump_t0` | 0.036 |
| `volatility` | 0.034 |
| `max_ratio` | 0.027 |
| `record_24m` | 0.020 |
| `ratio_now` | 0.014 |
| `group` | 0.013 |
| `base` | 0.006 |
| `seasonality` | -0.002 |
