"""Tests de l'API (sans réseau : la courbe vient du TSV).  Lancement : pytest -q"""
import sys
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
import api  # noqa: E402
from features import FEATURES  # noqa: E402
from scan import PageNotFound, month_index  # noqa: E402

client = TestClient(api.app)
TRAIN = pd.read_csv(ROOT / 'data' / 'train_v1_t0_plus_4mois.csv', dtype={'t0': str})
SERIES = pd.read_csv(ROOT / 'data' / 'wiki_series_2018_2026.tsv', sep='\t').set_index('title')


@pytest.fixture
def offline(monkeypatch):
    """Remplace le téléchargement par les séries du TSV."""
    def fake_fetch(title, last):
        if title not in SERIES.index:
            raise PageNotFound(title)
        return SERIES.loc[title, [str(i) for i in range(month_index(last) + 1)]].values.astype(float)
    monkeypatch.setattr(api, 'fetch_series', fake_fetch)


def test_health():
    r = client.get('/health')
    assert r.status_code == 200
    j = r.json()
    assert j['status'] == 'ok' and j['k'] == 4 and j['model_version'] == 'v1'


def test_predict_features_matches_model():
    rows = TRAIN.iloc[[0, 50, 120]]
    direct = api.MODEL['pipeline'].predict_proba(rows[FEATURES])[:, 1]
    for (_, x), p in zip(rows.iterrows(), direct):
        body = {c: (x[c] if c == 'group' else float(x[c])) for c in FEATURES}
        r = client.post('/predict_features', json=body)
        assert r.status_code == 200
        j = r.json()
        assert j['proba_tendance'] == pytest.approx(p, abs=1e-3)
        assert j['label'] == ('Tendance durable' if p >= 0.5 else 'Buzz')
        assert len(j['top_raisons']) == 3


def test_predict_features_bad_group():
    body = {c: 1.0 for c in FEATURES} | {'group': 'Inconnu'}
    assert client.post('/predict_features', json=body).status_code == 422


def test_predict_reproduces_training_case(offline):
    # Un cas d'entraînement dont t0 tombe dans la fenêtre d'un scan à fin 2023
    x = TRAIN[(TRAIN.t0 >= '202212') & (TRAIN.t0 <= '202308')].iloc[0]
    cat = SERIES.loc[x.title, 'category']
    r = client.post('/predict', params={'last': '202312', 'since': '202212'},
                    json={'title': x.title, 'category': cat})
    assert r.status_code == 200
    j = r.json()
    assert j['t0'] == x.t0
    for c in FEATURES:
        assert j['features'][c] == (x[c] if c == 'group' else pytest.approx(x[c]))
    assert j['verdict_verifiable_le'] and len(j['series']['views']) == month_index('202312') + 1


def test_predict_no_takeoff(offline):
    # Une page du TSV sans décollage dans la fenêtre d'un scan à fin 2023
    from features import find_t0
    li = month_index('202312')
    title = next(t for t in SERIES.index
                 if find_t0(SERIES.loc[t, [str(i) for i in range(li + 1)]].values.astype(float),
                            li - 12, li - 4) is None)
    r = client.post('/predict', params={'last': '202312', 'since': '202212'},
                    json={'title': title, 'category': SERIES.loc[title, 'category']})
    assert r.status_code == 200
    j = r.json()
    assert j['t0'] is None and j['label'] is None and 'décollage' in j['message']


def test_predict_not_found(offline):
    r = client.post('/predict', json={'title': 'Page qui n existe pas 123', 'category': 'Desserts'})
    assert r.status_code == 404


def test_predict_needs_category_or_group():
    r = client.post('/predict', json={'title': 'Vanilla'})
    assert r.status_code == 422


# ---------- Scan déclenché par l'API (faux scan, sans réseau ni écriture) ----------
def _wait_scan_done(timeout=5):
    import time
    t = time.time()
    while api.SCAN_STATE['running'] and time.time() - t < timeout:
        time.sleep(0.05)
    assert not api.SCAN_STATE['running']


def test_scan_run_and_status(monkeypatch):
    def fake_scan(last, since=None, progress=None):
        progress(1, 1)
        df = TRAIN.head(3).assign(exclu_raison=None, label='Buzz')
        return df, []
    monkeypatch.setattr(api, 'run_scan', fake_scan)
    r = client.post('/scan/run', params={'last': '202312'})
    assert r.status_code == 202
    _wait_scan_done()
    s = client.get('/scan/status').json()
    assert s['error'] is None and s['result']['decollages'] == 3 and s['last_month'] == '202312'


def test_scan_run_conflict_and_errors(monkeypatch):
    import threading
    gate = threading.Event()

    def slow_scan(last, since=None, progress=None):
        gate.wait(5)
        raise RuntimeError("scan incomplet")
    monkeypatch.setattr(api, 'run_scan', slow_scan)
    assert client.post('/scan/run').status_code == 202
    assert client.post('/scan/run').status_code == 409          # déjà en cours
    gate.set()
    _wait_scan_done()
    assert 'incomplet' in client.get('/scan/status').json()['error']
    assert client.post('/scan/run', params={'last': '2026-10'}).status_code == 422
    assert client.post('/scan/run', params={'last': '209912'}).status_code == 422
