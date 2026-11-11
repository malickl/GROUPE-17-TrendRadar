"""Étape 3 : API buzz / tendance durable.

Lancement (depuis la racine du projet) :  uvicorn src.api:app --reload
Documentation : http://127.0.0.1:8000/docs

Mise à jour du scan :
  - à la demande : POST /scan/run (suivi : GET /scan/status) ;
  - automatique : toutes les SCAN_AUTO_HOURS heures (variable d'environnement, défaut 24,
    0 = désactivé), l'API vérifie s'il existe un scan pour le dernier mois complet et
    le lance s'il manque.  Ex. : SCAN_AUTO_HOURS=6 uvicorn src.api:app
"""
import math
import os
import re
import sys
import threading
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
from features import FEATURES, GRP, K  # noqa: E402
from scan import (MODEL_PATH, OUT_DIR, SOURCE, TRAIN_CSV, PageNotFound,  # noqa: E402
                  analyse, fetch_series, index_month, label_of, last_complete_month,
                  month_add, predict, window)
from scan import scan as run_scan  # noqa: E402

MODEL = joblib.load(MODEL_PATH)
GROUPS = sorted(set(GRP.values()))

# Médianes de chaque colonne chez les Buzz et chez les Tendances (pour les phrases d'explication)
_T = pd.read_csv(TRAIN_CSV)
MEDIANS = {lab: _T[_T.label == lab][FEATURES[1:]].median().round(2).to_dict()
           for lab in ('Buzz', 'Tendance durable')}

# ---------- Scan en arrière-plan ----------
SCAN_AUTO_HOURS = float(os.getenv('SCAN_AUTO_HOURS', '24'))
_scan_lock = threading.Lock()
SCAN_STATE = {'running': False, 'trigger': None, 'last_month': None, 'since': None,
              'started_at': None, 'finished_at': None, 'progress': None,
              'result': None, 'error': None}


def _now():
    return datetime.now().isoformat(timespec='seconds')


def start_scan(last=None, since=None, trigger='manuel'):
    """Lance un scan dans un thread. Renvoie False si un scan tourne déjà."""
    if not _scan_lock.acquire(blocking=False):
        return False
    last = last or last_complete_month()
    SCAN_STATE.update(running=True, trigger=trigger, last_month=last, since=since,
                      started_at=_now(), finished_at=None, progress=None,
                      result=None, error=None)

    def work():
        try:
            df, errors = run_scan(
                last, since=since,
                progress=lambda k, n: SCAN_STATE.update(progress=f'{k}/{n} pages'))
            ok = df.exclu_raison.isna()
            SCAN_STATE['result'] = {
                'file': f'scan_{last}.csv', 'decollages': int(len(df)), 'valides': int(ok.sum()),
                'tendance': int((df.label[ok] == 'Tendance durable').sum()),
                'buzz': int((df.label[ok] == 'Buzz').sum()),
                'pages_en_erreur': len(errors)}
        except Exception as e:  # noqa: BLE001
            SCAN_STATE['error'] = str(e)
        finally:
            SCAN_STATE.update(running=False, finished_at=_now())
            _scan_lock.release()

    threading.Thread(target=work, daemon=True, name='scan').start()
    return True


def _auto_loop(stop):
    while not stop.is_set():
        if not (OUT_DIR / f'scan_{last_complete_month()}.csv').exists():
            start_scan(trigger='auto')
        stop.wait(SCAN_AUTO_HOURS * 3600)


@asynccontextmanager
async def lifespan(_app):
    stop = threading.Event()
    if SCAN_AUTO_HOURS > 0:
        threading.Thread(target=_auto_loop, args=(stop,), daemon=True, name='scan-auto').start()
    yield
    stop.set()


app = FastAPI(
    lifespan=lifespan,
    title="Radar de tendances : buzz ou tendance durable ?",
    version=MODEL['version'],
    description=(f"Prédit, {K} mois après le décollage d'un sujet sur Wikipédia (t0), "
                 "s'il sera encore là 2 ans plus tard (tendance durable) ou retombé (buzz). "
                 f"Source : {SOURCE}."),
)

# CORS : autorise une interface web hébergée ailleurs à appeler l'API depuis le navigateur.
# CORS_ORIGINS = liste séparée par des virgules (défaut "*" = toutes les origines).
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.getenv('CORS_ORIGINS', '*').split(',')],
    allow_methods=['*'], allow_headers=['*'],
)


# ---------- Schémas ----------
class PredictIn(BaseModel):
    title: str = Field(..., examples=['Dubai chocolate'], description="Titre de la page Wikipédia anglaise")
    category: Optional[str] = Field(None, examples=['Desserts'],
                                    description=f"Catégorie Wikipédia : {', '.join(GRP)}")
    group: Optional[str] = Field(None, description=f"Groupe du modèle (prioritaire) : {', '.join(GROUPS)}")


class FeaturesIn(BaseModel):
    group: str = Field(..., examples=['Desserts'])
    base: float = Field(..., examples=[120])
    record_24m: float = Field(..., examples=[150])
    ratio_now: float = Field(..., examples=[3.2])
    max_ratio: float = Field(..., examples=[4.1])
    jump_t0: float = Field(..., examples=[2.5])
    growth_since_t0: float = Field(..., examples=[1.6])
    volatility: float = Field(..., examples=[0.3])
    seasonality: float = Field(..., examples=[0.1])


# ---------- Explications ----------
def _n(x, d=1):
    """Nombre au format français (virgule)."""
    return f"{x:,.{d}f}".replace(',', ' ').replace('.', ',')


def sentence(col, x):
    if col == 'growth_since_t0':
        return (f"l'intérêt a encore augmenté de x{_n(x)} depuis le décollage" if x >= 1
                else f"l'intérêt a baissé depuis le décollage (x{_n(x)})")
    if col == 'jump_t0':
        return f"au décollage, l'intérêt a été multiplié par {_n(x)} en un mois"
    if col == 'volatility':
        return (f"la courbe est {'très irrégulière' if x >= 0.5 else 'irrégulière' if x >= 0.3 else 'régulière'}"
                f" depuis le décollage (volatilité {_n(x, 2)})")
    if col == 'max_ratio':
        return f"le pic atteint {_n(x)} fois l'ancien record"
    if col == 'ratio_now':
        return f"{K} mois après le décollage, l'intérêt est à {_n(x)} fois l'ancien record"
    if col == 'record_24m':
        return f"le record des 2 ans précédents était de {_n(x, 0)} vues par mois"
    if col == 'base':
        return f"le niveau habituel avant le décollage était de {_n(x, 0)} vues par mois"
    if col == 'seasonality':
        return f"ressemblance entre l'année -1 et l'année -2 : {_n(x, 2)}"
    return f"{col} = {x}"


def top_raisons(feats, n=3):
    """Les n variables les plus importantes du modèle, en phrases simples, avec
    l'indication de ce qu'elles suggèrent (comparaison aux médianes d'entraînement)."""
    out = []
    for col in [c for c in MODEL['importances'] if c != 'group'][:n]:
        x = feats[col]
        mb, mt = MEDIANS['Buzz'][col], MEDIANS['Tendance durable'][col]
        hint = 'plutôt signe de tendance' if abs(x - mt) < abs(x - mb) else 'plutôt signe de buzz'
        out.append({'variable': col, 'valeur': x, 'phrase': f"{sentence(col, x)} : {hint}",
                    'mediane_buzz': mb, 'mediane_tendance': mt})
    return out


def _predict_one(feats):
    p = float(predict(MODEL, [feats])[0])
    return round(p, 3), label_of(p)


# ---------- Routes ----------
@app.get('/', include_in_schema=False)
def root():
    """L'adresse de base redirige vers la documentation interactive."""
    return RedirectResponse('/docs')


@app.get('/ui', include_in_schema=False)
def ui():
    """Interface de démo : bulles vertes (tendance) et rouges (buzz)."""
    return FileResponse(ROOT / 'web' / 'index.html')


@app.get('/health')
def health():
    return {'status': 'ok', 'model_version': MODEL['version'], 'k': MODEL['k'],
            'model': MODEL.get('model_name')}


@app.post('/predict')
def predict_title(body: PredictIn, last: Optional[str] = None, since: Optional[str] = None):
    """Télécharge la courbe Wikipédia, cherche un décollage entre `since` (défaut : les
    12 derniers mois) et le dernier mois complet - 4, puis prédit.
    `last` (AAAAMM, optionnel) = dernier mois complet à utiliser."""
    if body.group:
        if body.group not in GROUPS:
            raise HTTPException(422, f"group inconnu. Valeurs possibles : {GROUPS}")
        group = body.group
    elif body.category in GRP:
        group = GRP[body.category]
    else:
        raise HTTPException(422, f"Indiquer 'category' ({list(GRP)}) ou 'group' ({GROUPS}).")

    last = last or last_complete_month()
    try:
        v = fetch_series(body.title, last)
    except PageNotFound:
        raise HTTPException(404, f"Page Wikipédia anglaise introuvable : « {body.title} »")
    except RuntimeError as e:
        raise HTTPException(503, f"Wikipédia ne répond pas : {e}")

    series = {'months': [index_month(i) for i in range(len(v))], 'views': [int(x) for x in v]}
    base = {'title': body.title, 'group': group, 'source': SOURCE, 'last_month': last,
            'series': series}

    a = analyse(v, last, since)
    if a is None:
        lo, hi = window(last, since)
        return {**base, 't0': None, 'label': None, 'proba_tendance': None,
                'message': (f"Pas de décollage détecté entre {lo} et {hi} : il faut un mois "
                            f"avec au moins 1,5x le record des 2 ans d'avant, et {K} mois "
                            "de recul.")}

    feats = {'group': group, **a['features']}
    proba, label = _predict_one(feats)
    out = {**base, 't0': a['t0'], 'label': label, 'proba_tendance': proba,
           'features': feats, 'top_raisons': top_raisons(feats),
           'verdict_verifiable_le': month_add(a['t0'], 24)}
    if a['exclu_raison']:
        out['avertissement'] = (f"Cas hors du périmètre du scan ({a['exclu_raison']}) : "
                                "prédiction moins fiable.")
    return out


@app.post('/predict_features')
def predict_features(body: FeaturesIn):
    """Prédiction à partir des 9 colonnes déjà calculées."""
    feats = body.model_dump()
    if feats['group'] not in GROUPS:
        raise HTTPException(422, f"group inconnu. Valeurs possibles : {GROUPS}")
    proba, label = _predict_one(feats)
    return {'label': label, 'proba_tendance': proba, 'features': feats,
            'top_raisons': top_raisons(feats)}


@app.post('/scan/run', status_code=202)
def scan_run(last: Optional[str] = None, since: Optional[str] = None):
    """Lance un nouveau scan en arrière-plan (quelques minutes). Suivi : GET /scan/status.
    `last` (AAAAMM, défaut : dernier mois complet), `since` (AAAAMM, défaut : last-12).
    Le résultat remplace outputs/scan_{last}.csv ; /scan/latest le sert dès qu'il est prêt."""
    for name, val in (('last', last), ('since', since)):
        if val is not None and not re.fullmatch(r'20\d{2}(0[1-9]|1[0-2])', val):
            raise HTTPException(422, f"{name} doit être au format AAAAMM")
    if last and last > last_complete_month():
        raise HTTPException(422, f"last doit être <= {last_complete_month()} (dernier mois complet)")
    if not start_scan(last, since, trigger='manuel'):
        raise HTTPException(409, "Un scan est déjà en cours : voir GET /scan/status")
    return {'status': 'lancé', 'last_month': SCAN_STATE['last_month'],
            'suivi': '/scan/status'}


@app.get('/scan/status')
def scan_status():
    """État du scan en cours ou du dernier scan lancé par l'API."""
    files = sorted(OUT_DIR.glob('scan_*.csv'))
    return {**SCAN_STATE,
            'auto': {'actif': SCAN_AUTO_HOURS > 0, 'verification_toutes_les_heures': SCAN_AUTO_HOURS},
            'dernier_fichier': files[-1].name if files else None,
            'dernier_mois_complet': last_complete_month()}


@app.get('/scan/latest')
def scan_latest():
    """Contenu du scan le plus récent (outputs/scan_AAAAMM.csv)."""
    files = sorted(OUT_DIR.glob('scan_*.csv'))
    if not files:
        raise HTTPException(404, "Aucun scan disponible : lancer python src/scan.py")
    f = files[-1]
    df = (pd.read_csv(f, dtype={'t0': str, 'verdict_verifiable_le': str})
            .rename(columns={'prediction': 'label'}))   # anciens fichiers
    rows = [{k: (None if isinstance(x, float) and math.isnan(x) else x) for k, x in r.items()}
            for r in df.to_dict(orient='records')]
    return {'file': f.name, 'last_month': f.stem.split('_')[1], 'n': len(rows), 'rows': rows}
