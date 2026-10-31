from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np
from .features import extract, FEATURE_NAMES, LOG_INDICES

WINDOW = 16
HORIZON = 10
TRAIN_IDS = ['Bearing1_1', 'Bearing1_2', 'Bearing1_3']
VAL_IDS = ['Bearing1_4']
TEST_IDS = ['Bearing1_5']

def prepare(raw_root: Path, out_root: Path):
    manifest = json.loads((raw_root / 'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('source_kind') != 'real_accelerated_bearing_experiment':
        raise ValueError('Real-data preparation requires the verified acquisition manifest.')
    out_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for record in manifest['records']:
        path = raw_root / record['path']
        body = path.read_bytes()
        if hashlib.sha256(body).hexdigest() != record['sha256']: raise ValueError(f'Checksum mismatch: {path}')
        with path.open(encoding='utf-8-sig') as f:
            header = f.readline().strip().split(',')
        if header != ['Horizontal_vibration_signals', 'Vertical_vibration_signals']:
            raise ValueError(f'Unverified channel order: {path}')
        capture = np.loadtxt(path, delimiter=',', skiprows=1)
        features = extract(capture)
        rows.append({'bearing_id': record['bearing_id'], 'capture_index': record['capture_index'],
                     'elapsed_seconds': (record['capture_index'] - 1) * 60,
                     'speed_rpm': manifest['speed_rpm'], 'load_kn': manifest['load_kn'],
                     'features': features.tolist()})
    for bearing, count in manifest['counts'].items():
        indices = sorted(r['capture_index'] for r in rows if r['bearing_id'] == bearing)
        if indices != list(range(1, count+1)): raise ValueError(f'Missing/duplicate captures: {bearing}')
    result = {'schema_version': 1, 'feature_names': FEATURE_NAMES, 'provenance': manifest,
              'rows': sorted(rows, key=lambda r: (r['bearing_id'], r['capture_index']))}
    (out_root / 'features.json').write_text(json.dumps(result, separators=(',', ':')), encoding='utf-8')
    return result

def windows(dataset, ids):
    xs, ys, metadata = [], [], []
    for bearing in ids:
        rows = sorted([r for r in dataset['rows'] if r['bearing_id'] == bearing], key=lambda r: r['capture_index'])
        if not rows: raise ValueError(f'Missing bearing: {bearing}')
        endpoint = rows[-1]['capture_index']
        for end in range(WINDOW-1, len(rows)-1):  # endpoint itself is never a future-warning target
            chunk = rows[end-WINDOW+1:end+1]
            if any(b['capture_index']-a['capture_index'] != 1 or b['elapsed_seconds']-a['elapsed_seconds'] != 60
                   for a,b in zip(chunk, chunk[1:])): continue
            x = np.array([r['features'] + [r['speed_rpm'], r['load_kn']] for r in chunk], dtype=np.float32)
            remaining = endpoint - rows[end]['capture_index']
            xs.append(x); ys.append(int(0 < remaining <= HORIZON))
            metadata.append({'bearing_id': bearing, 'capture_index': rows[end]['capture_index'],
                             'elapsed_seconds': rows[end]['elapsed_seconds'], 'endpoint_index': endpoint})
    if not xs: raise ValueError('No eligible complete windows.')
    return np.stack(xs), np.array(ys, dtype=np.int64), metadata

def compress(x):
    z = np.array(x, dtype=np.float64, copy=True)
    z[..., LOG_INDICES] = np.log1p(z[..., LOG_INDICES])
    return z

def fit_scaler(training_rows):
    z = compress(training_rows)
    mean, scale = z.mean(axis=0), z.std(axis=0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    return {'mean': mean.tolist(), 'scale': scale.tolist(),
            'reference': np.median(np.asarray(training_rows), axis=0).tolist()}

def transform(x, scaler):
    return ((compress(x) - np.array(scaler['mean'])) / np.array(scaler['scale'])).astype(np.float32)
