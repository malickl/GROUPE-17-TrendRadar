"""Protocole de validation sur 30 cas de référence.

Lancement (depuis la racine du projet) :  python src/validate.py
Sorties : data/reference_30.csv  et  models/validation_30.md

Protocole :
1. Les 30 cas sont tirés au hasard (graine fixe), AVANT de regarder les prédictions :
   15 Tendance durable + 15 Buzz, répartis sur les familles (group).
2. Chaque cas est prédit par un modèle qui ne l'a PAS vu à l'entraînement
   (validation croisée 10 plis : le cas est dans le pli de test). Même pipeline que train.py.
3. On compare à la vraie étiquette et on vérifie les critères ci-dessous.
"""
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
from features import FEATURES  # noqa: E402
from train import CANDIDATES, TRAIN_CSV, make_pipeline  # noqa: E402

REF_CSV = ROOT / 'data' / 'reference_30.csv'
REPORT = ROOT / 'models' / 'validation_30.md'
SEED = 2026
N_PAR_CLASSE = 15
CRITERES = {'precision_min': 0.75, 'auc_min': 0.80}
MODELE = 'Random Forest'


def tirer_references(df):
    """15 cas par étiquette, en tournant sur les familles pour les représenter toutes."""
    rng = random.Random(SEED)
    choisis = []
    for label in ['Tendance durable', 'Buzz']:
        par_famille = {g: list(s.index) for g, s in df[df.label == label].groupby('group')}
        for idx in par_famille.values():
            rng.shuffle(idx)
        familles = sorted(par_famille)
        rng.shuffle(familles)
        pris = []
        while len(pris) < N_PAR_CLASSE:
            for g in familles:
                if par_famille[g] and len(pris) < N_PAR_CLASSE:
                    pris.append(par_famille[g].pop())
        choisis += pris
    return sorted(choisis)


def probas_hors_entrainement(df):
    X, y = df[FEATURES], (df.label == 'Tendance durable').astype(int)
    p = np.zeros(len(df))
    for tr, te in StratifiedKFold(10, shuffle=True, random_state=0).split(X, y):
        m = make_pipeline(CANDIDATES[MODELE]()).fit(X.iloc[tr], y.iloc[tr])
        p[te] = m.predict_proba(X.iloc[te])[:, 1]
    return p


def main():
    df = pd.read_csv(TRAIN_CSV, dtype={'t0': str})
    df['proba_tendance'] = probas_hors_entrainement(df)
    df['prediction'] = np.where(df.proba_tendance >= 0.5, 'Tendance durable', 'Buzz')
    df['correct'] = df.prediction == df.label

    ref = df.loc[tirer_references(df),
                 ['title', 'group', 't0', 'label', 'proba_tendance', 'prediction', 'correct']]
    ref['proba_tendance'] = ref.proba_tendance.round(3)
    ref.to_csv(REF_CSV, index=False)

    y = (ref.label == 'Tendance durable').astype(int)
    acc = accuracy_score(ref.label, ref.prediction)
    auc = roc_auc_score(y, ref.proba_tendance)
    tn, fp, fn, tp = confusion_matrix(y, (ref.prediction == 'Tendance durable').astype(int)).ravel()
    acc_all = df.correct.mean()
    auc_all = roc_auc_score((df.label == 'Tendance durable').astype(int), df.proba_tendance)
    ok_acc, ok_auc = acc >= CRITERES['precision_min'], auc >= CRITERES['auc_min']
    ok_hasard = acc > 0.5

    print(f"30 cas de référence : précision {acc:.1%}, AUC {auc:.3f}")
    print(f"Tous les 175 cas (hors entraînement) : précision {acc_all:.1%}, AUC {auc_all:.3f}")

    v = lambda b: '✅' if b else '❌'  # noqa: E731
    lignes = [
        "# Validation sur 30 cas de référence", "",
        "## Protocole", "",
        f"1. **Choix des cas** : 30 cas tirés au hasard (graine {SEED}) parmi les 175 cas étiquetés, "
        f"{N_PAR_CLASSE} Tendance durable et {N_PAR_CLASSE} Buzz, répartis sur toutes les familles. "
        "Tirage fait avant de regarder les prédictions.",
        "2. **Prédiction honnête** : chaque cas est prédit par un modèle qui ne l'a pas vu à "
        "l'entraînement (validation croisée 10 plis, même pipeline que le modèle publié).",
        "3. **Comparaison** à la vraie étiquette (définie dans `features.py`).", "",
        "Fichier des cas : `data/reference_30.csv`. Relancer : `python src/validate.py`.", "",
        "## Critères de validation", "",
        "| Critère | Seuil | Résultat | Validé |", "|---|---|---|---|",
        f"| Battre le hasard (50 % sur 15/15) | > 50 % | {acc:.0%} | {v(ok_hasard)} |",
        f"| Précision | ≥ {CRITERES['precision_min']:.0%} | {acc:.0%} | {v(ok_acc)} |",
        f"| AUC | ≥ {CRITERES['auc_min']:.2f} | {auc:.2f} | {v(ok_auc)} |", "",
        "> **À lire avec prudence : l'échantillon est trop petit.** Avec 30 cas, une erreur "
        "pèse plus de 3 points et la marge d'erreur est d'environ ±15 points. Ces 30 cas servent "
        "surtout à montrer concrètement, cas par cas, où le modèle a raison et où il se trompe. "
        f"La mesure fiable est celle sur les 175 cas, avec la même méthode : précision "
        f"{acc_all:.0%}, AUC {auc_all:.2f} (voir plus bas et `report_v1.md`). Les seuils "
        "ci-dessus ont été fixés avant de voir les résultats et n'ont pas été modifiés.", "",
        "## Détail des erreurs", "",
        "| | Prédit Tendance | Prédit Buzz |", "|---|---|---|",
        f"| **Vraie Tendance** | {tp} | {fn} |",
        f"| **Vrai Buzz** | {fp} | {tn} |", "",
        "## Pour comparaison : les 175 cas", "",
        f"Même méthode sur les 175 cas : précision {acc_all:.0%}, AUC {auc_all:.2f}. "
        "Avec 30 cas, la marge d'erreur est d'environ ±15 points.", "",
        "## Les 30 cas", "",
        "| Sujet | Famille | t0 | Vraie réponse | Proba tendance | Prédiction | OK |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in ref.itertuples():
        lignes.append(f"| {r.title} | {r.group} | {r.t0} | {r.label} | {r.proba_tendance:.2f} "
                      f"| {r.prediction} | {v(r.correct)} |")
    REPORT.write_text("\n".join(lignes) + "\n", encoding='utf-8')
    print(f"Écrit : {REF_CSV.relative_to(ROOT)}, {REPORT.relative_to(ROOT)}")


if __name__ == '__main__':
    main()
