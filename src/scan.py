"""Étape 2 : scanner Wikipédia pour trouver les sujets qui décollent et les juger.

Exemples (depuis la racine du projet) :
  python src/scan.py                         # scan réel (télécharge, avec cache disque)
  python src/scan.py --asof 202312 --offline --check
        # contrôle : se placer fin 2023 avec les séries du TSV et comparer à l'entraînement
Sortie : outputs/scan_AAAAMM.csv  (AAAAMM = dernier mois complet utilisé)

Confidentialité : aucune mention du client dans les requêtes (User-Agent générique).
"""
import argparse
import json
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path
from urllib.parse import quote

import joblib
import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
from features import FEATURES, GRP, K, compute_features, exclusion_reason, find_t0  # noqa: E402

SERIES_TSV = ROOT / 'data' / 'wiki_series_2018_2026.tsv'
EXCLUSIONS = ROOT / 'data' / 'exclusions.txt'
TRAIN_CSV = ROOT / 'data' / 'train_v1_t0_plus_4mois.csv'
MODEL_PATH = ROOT / 'models' / 'model_v1.joblib'
CACHE_DIR = ROOT / 'cache' / 'pageviews'
OUT_DIR = ROOT / 'outputs'

START = '201801'
USER_AGENT = 'TrendRadarBot/0.1 (contact: mahassadiadou@gmail.com)'
BASE = ('https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/'
        'all-access/user/{title}/')
API = BASE + 'monthly/{start}0100/{end}0100'
API_DAILY = BASE + 'daily/{end}0100/{end}{lastday}00'
MAX_WORKERS = 8
SOURCE = 'Wikipedia pageviews (en, user)'


# ---------- Mois ----------
def month_add(yyyymm, n):
    y, m = int(yyyymm[:4]), int(yyyymm[4:])
    i = y * 12 + (m - 1) + n
    return f'{i // 12}{i % 12 + 1:02d}'


def month_index(yyyymm):
    """Index du mois dans la courbe (0 = 2018-01)."""
    return (int(yyyymm[:4]) - 2018) * 12 + int(yyyymm[4:]) - 1


def index_month(i):
    return month_add(START, i)


def last_complete_month(today=None):
    today = today or date.today()
    return month_add(f'{today.year}{today.month:02d}', -1)   # on exclut le mois en cours


# ---------- Téléchargement ----------
class PageNotFound(Exception):
    pass


_local = threading.local()


def _session():
    if not hasattr(_local, 's'):
        _local.s = requests.Session()
        _local.s.headers['User-Agent'] = USER_AGENT
    return _local.s


def _cache_path(title):
    return CACHE_DIR / f"{quote(title.replace(' ', '_'), safe='')}.json"


def _get(url, title, retries=6):
    """GET avec backoff exponentiel (429 / 5xx / réseau). Renvoie la liste 'items'
    (vide si 404 = aucune donnée sur la période)."""
    r = None
    for i in range(retries):
        try:
            r = _session().get(url, timeout=30)
        except requests.RequestException:
            r = None
        if r is not None and r.status_code == 200:
            return r.json().get('items', [])
        if r is not None and r.status_code == 404:
            return None
        wait = 2 ** i + random.random()
        if r is not None and r.headers.get('Retry-After', '').isdigit():
            wait = max(wait, int(r.headers['Retry-After']))
        time.sleep(wait)
    raise RuntimeError(f"échec du téléchargement après {retries} essais : {title}"
                       f" (dernier code : {getattr(r, 'status_code', 'réseau')})")


def fetch_views(title, last):
    """Vues mensuelles {AAAAMM: vues} de START à `last` inclus. Cache disque.
    Lève PageNotFound si Wikipédia ne connaît pas la page.

    Le dernier mois est recalculé depuis les vues JOURNALIÈRES : l'agrégat mensuel
    de Wikimedia peut être encore incomplet quelques jours après la fin du mois
    (constaté : 93 vues au lieu de 3 235 pour Cyclamate en 2026-10)."""
    path = _cache_path(title)
    if path.exists():
        c = json.loads(path.read_text())
        if c.get('not_found'):
            raise PageNotFound(title)
        if c['end'] >= last:
            return {m: v for m, v in c['views'].items() if m <= last}

    t = quote(title.replace(' ', '_'), safe='')
    items = _get(API.format(title=t, start=START, end=last), title)
    if items is None:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({'title': title, 'not_found': True}))
        raise PageNotFound(title)
    views = {it['timestamp'][:6]: it['views'] for it in items}

    lastday = (pd.Period(f'{last[:4]}-{last[4:]}', 'M').days_in_month)
    daily = _get(API_DAILY.format(title=t, end=last, lastday=lastday), title)
    if daily:
        views[last] = sum(it['views'] for it in daily)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({'title': title, 'end': last, 'views': views}))
    return views


def series_from_views(views, last):
    """Courbe sans trou de START à `last` ; mois absents = 0."""
    return np.array([float(views.get(index_month(i), 0)) for i in range(month_index(last) + 1)])


def fetch_series(title, last):
    return series_from_views(fetch_views(title, last), last)


# ---------- Analyse d'une courbe ----------
def window(last, since=None):
    """Fenêtre des t0 jugeables : de `since` à last-K (il faut K mois de recul).
    Par défaut, `since` = last-12 : décollages des 12 derniers mois
    (ex. last=202610 -> t0 de 202510 à 202606)."""
    return since or month_add(last, -12), month_add(last, -K)


def analyse(v, last, since=None):
    """Cherche un décollage jugeable dans la fenêtre `window(last, since)`.
    Renvoie None s'il n'y en a pas, sinon dict(t0, features, exclu_raison)."""
    lo, hi = window(last, since)
    t0 = find_t0(v, month_index(lo), month_index(hi))
    if t0 is None:
        return None
    return {'t0': index_month(t0),
            'features': compute_features(v, t0),
            'exclu_raison': exclusion_reason(v, t0)}


def predict(model, rows):
    """rows : liste de dicts contenant les 9 colonnes. Renvoie les probas de tendance."""
    X = pd.DataFrame(rows)[model['features']]
    return model['pipeline'].predict_proba(X)[:, 1]


def label_of(p):
    return 'Tendance durable' if p >= 0.5 else 'Buzz'


# ---------- Scan ----------
def load_pages():
    D = pd.read_csv(SERIES_TSV, sep='\t')
    excl = {l.strip() for l in EXCLUSIONS.read_text(encoding='utf-8').splitlines() if l.strip()}
    return D[~D.title.isin(excl)].reset_index(drop=True), len(excl)


MAX_ERROR_RATE = 0.05   # au-delà, le scan est jugé incomplet et n'écrit pas de fichier
OUT_COLS = (['title', 'group', 't0'] + FEATURES[1:] +
            ['proba_tendance', 'label', 'verdict_verifiable_le', 'exclu_raison'])


def get_series_all(pages, last, offline, progress=None):
    """{title: courbe}. offline = séries du TSV (coupées à `last`), sinon téléchargement.
    progress(fait, total) est appelé au fil du téléchargement (optionnel)."""
    if offline:
        n = month_index(last) + 1
        cols = [str(i) for i in range(n)]
        if cols[-1] not in pages.columns:
            raise ValueError(f"Le TSV s'arrête avant {last} : impossible en mode hors ligne.")
        return {t: pages.loc[i, cols].values.astype(float) for i, t in enumerate(pages.title)}, []

    out, errors = {}, []
    with ThreadPoolExecutor(MAX_WORKERS) as ex:
        futs = {ex.submit(fetch_series, t, last): t for t in pages.title}
        for k, f in enumerate(as_completed(futs), 1):
            t = futs[f]
            try:
                out[t] = f.result()
            except PageNotFound:
                errors.append((t, 'page introuvable'))
            except Exception as e:  # noqa: BLE001
                errors.append((t, str(e)))
            if progress:
                progress(k, len(futs))
            if k % 200 == 0 or k == len(futs):
                print(f"  {k}/{len(futs)} pages")
    return out, errors


def scan(last, offline=False, since=None, progress=None):
    """Scanne toutes les pages et écrit outputs/scan_{last}.csv. Renvoie (df, erreurs).
    Lève RuntimeError (sans écrire) si trop de pages n'ont pas pu être téléchargées."""
    model = joblib.load(MODEL_PATH)
    pages, n_excl = load_pages()
    lo, hi = window(last, since)
    print(f"Dernier mois complet : {last} | décollages cherchés de {lo} à {hi} | "
          f"{len(pages)} pages ({n_excl} exclusions retirées)"
          f" | {'hors ligne (TSV)' if offline else 'téléchargement'}")
    series, errors = get_series_all(pages, last, offline, progress)
    failed = [e for e in errors if e[1] != 'page introuvable']
    if len(failed) > MAX_ERROR_RATE * len(pages):
        raise RuntimeError(f"{len(failed)} pages non téléchargées sur {len(pages)} "
                           f"(ex. {failed[0][1]}) : scan incomplet, aucun fichier écrit.")
    cat = dict(zip(pages.title, pages.category))

    rows = []
    for t, v in series.items():
        a = analyse(v, last, since)
        if a is None:
            continue
        rows.append({'title': t, 'group': GRP[cat[t]], 't0': a['t0'], **a['features'],
                     'exclu_raison': a['exclu_raison']})
    if rows:
        df = pd.DataFrame(rows)
        df['proba_tendance'] = np.round(predict(model, rows), 3)
        df['label'] = df.proba_tendance.map(label_of)
        df['verdict_verifiable_le'] = df.t0.map(lambda m: month_add(m, 24))
        # Cas valides d'abord, puis par probabilité décroissante
        df = (df[OUT_COLS].assign(_ex=df.exclu_raison.notna())
                .sort_values(['_ex', 'proba_tendance'], ascending=[True, False])
                .drop(columns='_ex').reset_index(drop=True))
    else:
        print("Aucun décollage détecté.")
        df = pd.DataFrame(columns=OUT_COLS)

    # Écriture atomique : /scan/latest ne lit jamais un fichier à moitié écrit
    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / f'scan_{last}.csv'
    tmp = out.with_suffix('.tmp')
    df.to_csv(tmp, index=False)
    tmp.replace(out)
    ok = df.exclu_raison.isna()
    print(f"{len(df)} décollages : {ok.sum()} valides "
          f"({(df.label[ok] == 'Tendance durable').sum()} tendance, "
          f"{(df.label[ok] == 'Buzz').sum()} buzz), {(~ok).sum()} exclus par les garde-fous")
    if errors:
        print(f"{len(errors)} pages en erreur, ex. : {errors[:3]}")
    print(f"Sortie : {out.relative_to(ROOT)}")
    return df, errors


def check_against_training(df, last, since=None):
    """Contrôle : les cas d'entraînement dont t0 tombe dans la fenêtre du scan
    doivent être retrouvés avec le même t0 et les mêmes colonnes."""
    T = pd.read_csv(TRAIN_CSV, dtype={'t0': str})
    lo, hi = window(last, since)
    T = T[(T.t0 >= lo) & (T.t0 <= hi)]
    S = df.set_index('title') if len(df) else pd.DataFrame()
    num = FEATURES[1:]
    found = same = 0
    diffs = []
    for _, x in T.iterrows():
        if x.title not in S.index:
            diffs.append((x.title, 'absent du scan'))
            continue
        found += 1
        s = S.loc[x.title]
        if s.t0 == x.t0 and s.group == x.group and all(abs(s[c] - x[c]) < 1e-6 for c in num):
            same += 1
        else:
            diffs.append((x.title, f"t0 {s.t0} vs {x.t0}"))
    print(f"\nContrôle : {len(T)} cas d'entraînement avec t0 entre {lo} et {hi} ; "
          f"{found} retrouvés, {same} identiques.")
    for d in diffs[:10]:
        print("  ", *d)
    return same == len(T)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--asof', help="dernier mois complet AAAAMM (défaut : le mois dernier)")
    ap.add_argument('--since', help="premier mois de décollage cherché AAAAMM "
                                    "(défaut : 12 mois avant le dernier mois complet)")
    ap.add_argument('--offline', action='store_true', help="utiliser les séries du TSV")
    ap.add_argument('--check', action='store_true', help="comparer aux cas d'entraînement")
    a = ap.parse_args()
    last = a.asof or last_complete_month()
    try:
        df, _ = scan(last, offline=a.offline, since=a.since)
    except (RuntimeError, ValueError) as e:
        sys.exit(str(e))
    if a.check:
        check_against_training(df, last, a.since)


if __name__ == '__main__':
    main()
