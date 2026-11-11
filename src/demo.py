"""Démo : enrichit le scan réel avec des champs FICTIFS (pays, veille, historique…).

Lancement :  python src/demo.py
Entrées : outputs/scan_202610.csv (réel) + data/demo_contenu.json (textes rédigés à la main)
Sortie  : data/demo_scan.json (fixe : graine aléatoire constante), servi par /scan/latest.

Les champs réels du modèle (title, group, t0, label, proba_tendance, variables, etc.)
ne sont jamais modifiés. Chaque sujet porte demo_fictif = true.
"""
import json
import random
import sys
from pathlib import Path
from urllib.parse import quote, quote_plus

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
from scan import month_add  # noqa: E402

SCAN_CSV = ROOT / 'outputs' / 'scan_202610.csv'
CONTENU = ROOT / 'data' / 'demo_contenu.json'
DEMO_JSON = ROOT / 'data' / 'demo_scan.json'
SEED = 2026
FIN = '202610'           # dernier mois de l'historique
DEBUT_PRED = '202601'    # date_prediction entre janvier et octobre 2026

CAPITALES = {  # code ISO : (ville, lat, lon)
    'FR': ('Paris', 48.8566, 2.3522), 'US': ('Washington', 38.9072, -77.0369),
    'GB': ('Londres', 51.5074, -0.1278), 'DE': ('Berlin', 52.5200, 13.4050),
    'IT': ('Rome', 41.9028, 12.4964), 'ES': ('Madrid', 40.4168, -3.7038),
    'BR': ('Brasília', -15.7939, -47.8828), 'MX': ('Mexico', 19.4326, -99.1332),
    'IN': ('New Delhi', 28.6139, 77.2090), 'JP': ('Tokyo', 35.6762, 139.6503),
    # pays d'origine supplémentaires
    'LB': ('Beyrouth', 33.8938, 35.5018), 'TW': ('Taipei', 25.0330, 121.5654),
    'BE': ('Bruxelles', 50.8503, 4.3517), 'YE': ('Sanaa', 15.3694, 44.1910),
    'PF': ('Papeete', -17.5516, -149.5585), 'NG': ('Abuja', 9.0765, 7.3986),
    'SE': ('Stockholm', 59.3293, 18.0686), 'MA': ('Rabat', 34.0209, -6.8416),
    'NL': ('Amsterdam', 52.3676, 4.9041), 'TR': ('Ankara', 39.9334, 32.8597),
    'ET': ('Addis-Abeba', 9.0300, 38.7400), 'AU': ('Canberra', -35.2809, 149.1300),
}
TIRAGE = ['FR', 'US', 'GB', 'DE', 'IT', 'ES', 'BR', 'MX', 'IN', 'JP']  # pays sans origine claire

# Pays d'origine des sujets quand il est clair ; les autres sont tirés au hasard dans TIRAGE.
ORIGINE = {
    'Maritozzo': 'IT', 'Halawet el jibn': 'LB', 'Wheel pie': 'TW', 'Antwerp hands': 'BE',
    'Polianthes tuberosa': 'MX', 'Khaliat nahl': 'YE', 'Ladyfingers (biscuits)': 'IT',
    'Frog eye salad': 'US', 'Monoï oil': 'PF', 'Khubani ka meetha': 'IN', 'Feijoa': 'BR',
    'Icebox cookie': 'US', '7 Up cake': 'US', 'Ogiri': 'NG', 'Açai bowl': 'BR',
    'Shahi tukra': 'IN', 'Budapest roll': 'SE', 'Char magaz': 'IN', 'Zobo (drink)': 'NG',
    'Shebakia': 'MA', 'Rijstevlaai': 'NL', 'Crisp (dessert)': 'US', 'Platonia': 'BR',
    'Nevzine tatlısı': 'TR', 'Chirote': 'IN', 'Ensete': 'ET', 'Solanum centrale': 'AU',
    'Diplomat pudding': 'FR', 'Advantame': 'JP', "Burry's": 'US',
    'Brown Betty (dessert)': 'US', 'Thekua': 'IN',
}

# Sujets retirés de la démo (hors sujet ou trop génériques pour un manager)
EXCLUS_DEMO = {'Mast seeding', 'Types of cocoa beans', 'Fragrance compound', 'Crazy Mountain'}


def months_between(a, b):
    out, m = [], a
    while m <= b:
        out.append(m)
        m = month_add(m, 1)
    return out


def sources(title):
    """Liens réels et fonctionnels (pas d'articles inventés)."""
    return [
        {'titre': f'Wikipédia : {title}',
         'url': f"https://en.wikipedia.org/wiki/{quote(title.replace(' ', '_'), safe='')}"},
        {'titre': f'Google Trends : intérêt de recherche pour « {title} »',
         'url': f'https://trends.google.com/trends/explore?date=today%2012-m&q={quote_plus(title)}'},
        {'titre': f'Google Actualités : articles sur « {title} »',
         'url': f'https://news.google.com/search?q={quote_plus(title)}'},
    ]


def historique(rng, mois, proba_finale, label):
    """Probas mensuelles qui convergent vers la proba actuelle, toujours du même côté
    de 0,5 que le verdict actuel (historique cohérent avec le label)."""
    tendance = label == 'Tendance durable'
    lo, hi = (0.51, 0.99) if tendance else (0.01, 0.49)
    depart = min(hi, max(lo, proba_finale + rng.uniform(-0.15, 0.15)))
    n = len(mois)
    out = []
    for i, m in enumerate(mois):
        if i == n - 1:
            p = proba_finale                       # le dernier mois = la vraie prédiction
        else:
            w = i / max(n - 1, 1)
            p = (1 - w) * depart + w * proba_finale + rng.uniform(-0.03, 0.03)
            p = min(hi, max(lo, p))
        out.append({'mois': m, 'label': label, 'proba_tendance': round(p, 3)})
    return out


def build():
    rng = random.Random(SEED)
    scan = pd.read_csv(SCAN_CSV, dtype={'t0': str, 'verdict_verifiable_le': str})
    scan = scan.rename(columns={'prediction': 'label'})
    # Démo : uniquement les cas valides (sans raison d'exclusion), moins les hors-sujet
    scan = scan[scan.exclu_raison.isna() & ~scan.title.isin(EXCLUS_DEMO)]
    contenu = json.loads(CONTENU.read_text(encoding='utf-8'))
    manquants = set(scan.title) - set(contenu)
    assert not manquants, f"Contenu manquant pour : {sorted(manquants)}"

    rows = []
    for r in scan.to_dict(orient='records'):
        r = {k: (None if isinstance(v, float) and pd.isna(v) else v) for k, v in r.items()}
        element, veille, matieres = contenu[r['title']]
        tirage = rng.choice(TIRAGE)          # toujours tiré, pour garder la suite aléatoire stable
        iso = ORIGINE.get(r['title'], tirage)
        _, lat, lon = CAPITALES[iso]
        debut = max(DEBUT_PRED, month_add(r['t0'], 1))   # postérieur à t0
        date_pred = rng.choice(months_between(debut, FIN))
        rows.append({
            **r,                                           # champs réels, inchangés
            'domaine': 'Parfum' if r['group'] == 'Olfactif' else 'Goût',
            'country': iso,
            'country_source': 'origine' if r['title'] in ORIGINE else 'aleatoire',
            'lat': round(lat + rng.uniform(-0.6, 0.6), 4),
            'lon': round(lon + rng.uniform(-0.6, 0.6), 4),
            'element_cle': element,
            'veille': {'resume': veille, 'sources': sources(r['title'])},
            'matieres_premieres': matieres,
            'date_prediction': date_pred,
            'historique': historique(rng, months_between(date_pred, FIN),
                                     r['proba_tendance'], r['label']),
            'demo_fictif': True,
        })
    out = {'file': DEMO_JSON.name, 'source_reelle': SCAN_CSV.name, 'last_month': FIN,
           'demo_fictif': True,
           'avertissement': ("Données de démonstration : domaine, country, lat, lon, element_cle, "
                             "veille, matieres_premieres, date_prediction et historique sont "
                             "fictifs (country = pays d'origine du sujet si country_source vaut "
                             "'origine', sinon tiré au hasard). title, group, t0, label, "
                             "proba_tendance et les variables du modèle sont réels. Seuls les "
                             "cas valides sont listés."),
           'n': len(rows), 'rows': rows}
    DEMO_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f"{len(rows)} sujets enrichis -> {DEMO_JSON.relative_to(ROOT)}")
    return out


if __name__ == '__main__':
    build()
