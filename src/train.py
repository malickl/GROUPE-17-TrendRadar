"""Étape 1 : entraîner le modèle buzz / tendance durable et l'exporter.

Lancement (depuis la racine du projet) :  python src/train.py
Sorties : models/model_v1.joblib  et  models/report_v1.md
"""
from datetime import date
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
from features import FEATURES, K  # noqa: E402  (règle d'or : une seule définition des colonnes)

TRAIN_CSV = ROOT / 'data' / 'train_v1_t0_plus_4mois.csv'
MODEL_PATH = ROOT / 'models' / 'model_v1.joblib'
REPORT_PATH = ROOT / 'models' / 'report_v1.md'
VERSION = 'v1'
NUM = [c for c in FEATURES if c != 'group']


def make_pipeline(clf):
    pre = ColumnTransformer([
        ('group', OneHotEncoder(handle_unknown='ignore'), ['group']),
        ('num', 'passthrough', NUM),
    ])
    return Pipeline([('pre', pre), ('clf', clf)])


CANDIDATES = {
    'Random Forest': lambda: RandomForestClassifier(
        n_estimators=500, min_samples_leaf=2, random_state=0, n_jobs=-1),
    'Gradient Boosting': lambda: HistGradientBoostingClassifier(random_state=0),
}


def evaluate(X, y):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=0)
    scores = {}
    for name, make in CANDIDATES.items():
        r = cross_validate(make_pipeline(make()), X, y, cv=cv,
                           scoring=['accuracy', 'roc_auc'], n_jobs=1)
        scores[name] = {
            'accuracy': float(np.mean(r['test_accuracy'])),
            'accuracy_std': float(np.std(r['test_accuracy'])),
            'auc': float(np.mean(r['test_roc_auc'])),
            'auc_std': float(np.std(r['test_roc_auc'])),
        }
        s = scores[name]
        print(f"{name:18s} précision {s['accuracy']:.3f} (±{s['accuracy_std']:.3f})"
              f" | AUC {s['auc']:.3f} (±{s['auc_std']:.3f})")
    return scores


def held_out_importances(make, X, y):
    """Importance de chaque colonne d'origine ('group' regroupé) : baisse d'AUC quand on
    mélange la colonne, mesurée sur des cas NON vus (10 plis). Sur les données
    d'entraînement, la forêt les connaît par cœur et toutes les importances sont ~0."""
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    imps = []
    for tr, te in cv.split(X, y):
        p = make_pipeline(make()).fit(X.iloc[tr], y.iloc[tr])
        pi = permutation_importance(p, X.iloc[te], y.iloc[te], scoring='roc_auc',
                                    n_repeats=10, random_state=0)
        imps.append(pi.importances_mean)
    return (pd.Series(np.mean(imps, axis=0), index=FEATURES)
            .sort_values(ascending=False).round(4).to_dict())


def main():
    df = pd.read_csv(TRAIN_CSV)
    X = df[FEATURES]
    y = (df['label'] == 'Tendance durable').astype(int)
    print(f"{len(df)} cas : {int(y.sum())} Tendance durable, {int((1 - y).sum())} Buzz")
    print(f"Hasard (toujours Buzz) : {max(y.mean(), 1 - y.mean()):.3f}\n")

    print("Validation croisée 10 plis x 5 répétitions...")
    scores = evaluate(X, y)
    best = max(scores, key=lambda n: (scores[n]['auc'], scores[n]['accuracy']))
    print(f"\nModèle retenu : {best}")

    pipe = make_pipeline(CANDIDATES[best]()).fit(X, y)

    importances = held_out_importances(CANDIDATES[best], X, y)
    print("Top variables :", ", ".join(list(importances)[:3]))

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump({
        'pipeline': pipe,
        'features': FEATURES,
        'k': K,
        'model_name': best,
        'trained_on': {'file': TRAIN_CSV.name, 'n': len(df),
                       'n_tendance': int(y.sum()), 'date': date.today().isoformat()},
        'cv_scores': scores,
        'importances': importances,
        'classes': {0: 'Buzz', 1: 'Tendance durable'},
        'version': VERSION,
    }, MODEL_PATH)
    print(f"Modèle sauvegardé : {MODEL_PATH.relative_to(ROOT)}")

    lines = [
        f"# Rapport modèle {VERSION}", "",
        f"- Données : `{TRAIN_CSV.name}` : {len(df)} cas "
        f"({int(y.sum())} Tendance durable, {int((1 - y).sum())} Buzz)",
        f"- Prédiction à t0 + {K} mois",
        f"- Hasard (toujours Buzz) : {max(y.mean(), 1 - y.mean()):.0%}",
        f"- Entraîné le {date.today().isoformat()}", "",
        "## Validation croisée (10 plis x 5 répétitions)", "",
        "| Modèle | Précision | AUC |", "|---|---|---|",
    ]
    for n, s in scores.items():
        mark = " **(retenu)**" if n == best else ""
        lines.append(f"| {n}{mark} | {s['accuracy']:.1%} ± {s['accuracy_std']:.1%} "
                     f"| {s['auc']:.3f} ± {s['auc_std']:.3f} |")
    lines += ["", "## Variables les plus utiles", "",
              "Baisse d'AUC quand on mélange la colonne, mesurée sur des cas non vus "
              "(10 plis). Plus c'est haut, plus elle compte.", "",
              "| Variable | Importance |", "|---|---|"]
    lines += [f"| `{f}` | {v:.3f} |" for f, v in importances.items()]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding='utf-8')
    print(f"Rapport : {REPORT_PATH.relative_to(ROOT)}")


if __name__ == '__main__':
    main()
