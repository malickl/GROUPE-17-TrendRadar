"""Calcul commun des colonnes (features) — À UTILISER PARTOUT (entraînement, scan, API).

Une "courbe" = vues Wikipédia mensuelles (anglais, agent=user), une valeur par mois,
dans l'ordre chronologique, sans trou. Ce fichier reproduit EXACTEMENT
data/train_v1_t0_plus_4mois.csv (vérifié par verify_features() en bas).
"""
import numpy as np
import pandas as pd

K = 4                 # on prédit 4 mois après t0
LOOKBACK = 24         # t0 a besoin de 24 mois d'historique avant lui
FLOOR = 50            # plancher anti-bruit pour les petites pages

FEATURES = ['group', 'base', 'record_24m', 'ratio_now', 'max_ratio', 'jump_t0',
            'growth_since_t0', 'volatility', 'seasonality']

# Catégorie Wikipédia -> groupe (colonne 'group' du modèle)
GRP = {"Perfume ingredients": "Olfactif", "Essential oils": "Olfactif",
       "Sugar substitutes": "Arômes & sucre", "Flavors": "Arômes & sucre",
       "Spices": "Épices", "Edible nuts and seeds": "Noix & graines",
       "Tropical fruit": "Fruits tropicaux", "Non-alcoholic drinks": "Boissons",
       "Desserts": "Desserts"}

# Facteur de déclin moyen de chaque catégorie (utilisé seulement pour CRÉER les étiquettes)
DECLINE = {'Desserts': 0.939, 'Edible nuts and seeds': 0.83, 'Essential oils': 0.471,
           'Flavors': 0.562, 'Non-alcoholic drinks': 0.698, 'Perfume ingredients': 0.644,
           'Spices': 0.658, 'Sugar substitutes': 0.526, 'Tropical fruit': 0.829}


def roll(v):
    """Moyenne glissante 3 mois, NON centrée (mois courant + 2 précédents) -> pas de fuite du futur."""
    return pd.Series(v, dtype=float).rolling(3, min_periods=1).mean().values


def is_takeoff(v, t):
    """Règle de décollage au mois t : brut >= 1,5x le record des 24 mois d'avant
    ET moyenne 3 mois >= 1,2x ce record (record = max des moyennes 3 mois, plancher 50)."""
    rt = roll(v)
    rec = max(rt[t - LOOKBACK:t].max(), FLOOR)
    return v[t] >= 1.5 * rec and rt[t] >= 1.2 * rec


def find_t0(v, start, end):
    """Premier mois t dans [start, end] (indices inclus) qui respecte la règle de décollage."""
    for t in range(max(start, LOOKBACK), end + 1):
        if is_takeoff(v, t):
            return t
    return None


def exclusion_reason(v, t0, k=K):
    """Garde-fous du SCAN (6 des 175 cas d'entraînement ne les passeraient pas : normal,
    ce sont des filtres anti-faux-positifs, pas une condition du modèle).
    Renvoie None si le cas est valide, sinon la raison."""
    # NB : on GARDE les pages récentes (créées pour le buzz, ex. Grimace Shake) :
    # elles sont dans l'entraînement. Seuls les renommages suspects sont à vérifier à la main.
    if v[t0 - 12] >= 0.7 * v[t0]:
        return "saisonnier (même mois l'an passé déjà >=70%)"
    if roll(v)[t0:t0 + k + 1].max() < 500:
        return "trop petit (<500 vues/mois en moyenne 3 mois)"
    return None


def compute_features(v, t0, k=K):
    """Colonnes du modèle. N'utilise QUE les mois <= t0+k."""
    v = np.asarray(v, dtype=float)
    r = roll(v); end = t0 + k
    assert end < len(v), "pas assez de mois après t0"
    rec = r[t0 - LOOKBACK:t0].max(); win = v[t0:end + 1]
    a = v[t0 - 12:t0]; b = v[t0 - 24:t0 - 12]
    seas = np.corrcoef(a, b)[0, 1] if a.std() > 0 and b.std() > 0 else 0
    return dict(
        base=round(np.median(v[t0 - 24:t0])),                      # niveau "normal" avant
        record_24m=round(rec),                                      # meilleur niveau des 2 ans d'avant
        ratio_now=round(r[end] / max(rec, FLOOR), 2),               # niveau à t0+k vs ancien record
        max_ratio=round(r[t0:end + 1].max() / max(rec, FLOOR), 2),  # hauteur du pic vs record
        jump_t0=round(r[t0] / max(r[t0 - 1], 1), 2),                # brutalité du décollage
        growth_since_t0=round(r[end] / max(r[t0], 1), 2),           # continue-t-il de monter ?
        volatility=round(np.std(win) / max(np.mean(win), 1), 2),    # montagnes russes ?
        seasonality=round(seas, 2),                                 # l'an -1 ressemble à l'an -2 ?
    )


def make_label(v, t0, category, last12):
    """Étiquette (pour entraîner seulement) : last12 = moyenne des 12 derniers mois,
    à >= 24 mois après t0. Tendance si >= 0,7 ; Buzz si < 0,4 ; sinon exclu."""
    pk = roll(v)[t0:t0 + 13].max()
    pers = (last12 / pk) / DECLINE[category]
    return 'Tendance durable' if pers >= 0.7 else 'Buzz' if pers < 0.4 else None


def verify_features(series_tsv='data/wiki_series_2018_2026.tsv',
                    train_csv='data/train_v1_t0_plus_4mois.csv'):
    """Test de non-régression : recalcule t0 + colonnes pour les 175 cas et compare."""
    M = [f'{y}{m:02d}' for y in range(2018, 2027) for m in range(1, 13)][:106]
    D = pd.read_csv(series_tsv, sep='\t').set_index('title')
    T = pd.read_csv(train_csv)
    bad = 0
    for _, x in T.iterrows():
        v = D.loc[x.title, [str(i) for i in range(106)]].values.astype(float)
        t0 = find_t0(v, M.index('202001'), M.index('202312'))
        f = compute_features(v, t0)
        ok = M[t0] == str(x.t0) and all(abs(f[c] - x[c]) < 1e-6 for c in f)
        bad += not ok
    print(f"{len(T) - bad}/{len(T)} cas identiques")
    return bad == 0


if __name__ == '__main__':
    verify_features()
