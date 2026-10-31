"""Deterministic vibration concepts. Units follow source accelerometer channels."""
from __future__ import annotations
import numpy as np
from scipy import signal, stats

SAMPLE_RATE = 25600
SAMPLES = 32768
BAND_EDGES = np.array([0, 200, 500, 1000, 2000, 4000, 6400, 9600, 12800], dtype=float)
CHANNELS = ('horizontal', 'vertical')
FEATURE_NAMES = []
GROUPS = {'impulsiveness': [], 'amplitude': [], 'spectral_energy': [], 'coherence': [], 'relative_phase': []}
for channel in CHANNELS:
    for stat in ('rms', 'kurtosis', 'crest_factor'):
        GROUPS['amplitude' if stat == 'rms' else 'impulsiveness'].append(len(FEATURE_NAMES))
        FEATURE_NAMES.append(f'{channel}_{stat}')
    for band in range(8):
        GROUPS['spectral_energy'].append(len(FEATURE_NAMES))
        FEATURE_NAMES.append(f'{channel}_band_{band}_energy')
for band in range(8):
    GROUPS['coherence'].append(len(FEATURE_NAMES))
    FEATURE_NAMES.append(f'cross_band_{band}_coherence')
    for component in ('sin', 'cos'):
        GROUPS['relative_phase'].append(len(FEATURE_NAMES))
        FEATURE_NAMES.append(f'cross_band_{band}_phase_{component}')
N_FEATURES = len(FEATURE_NAMES)
PHASE_INDICES = GROUPS['relative_phase']
LOG_INDICES = GROUPS['amplitude'] + GROUPS['impulsiveness'] + GROUPS['spectral_energy']

def extract(capture, sample_rate=SAMPLE_RATE):
    x = np.asarray(capture, dtype=np.float64)
    if sample_rate != SAMPLE_RATE:
        raise ValueError('This feature schema requires 25600 Hz; resample with a validated external pipeline first.')
    if x.shape != (SAMPLES, 2):
        raise ValueError(f'Expected {SAMPLES} aligned samples and exactly two channels.')
    if not np.isfinite(x).all(): raise ValueError('Capture contains non-finite samples.')
    x = x - x.mean(axis=0)
    rms = np.sqrt(np.mean(x * x, axis=0))
    if (rms < 1e-10).any(): raise ValueError('Flat channel: no usable vibration signal.')
    kwargs = dict(fs=sample_rate, window='hann', nperseg=2048, noverlap=1024, detrend='constant')
    f, pxx = signal.welch(x[:, 0], **kwargs)
    _, pyy = signal.welch(x[:, 1], **kwargs)
    _, pxy = signal.csd(x[:, 0], x[:, 1], **kwargs)
    spectra = [pxx, pyy]
    df = f[1] - f[0]
    masks = [(f >= a) & (f < b if i < 7 else f <= b)
             for i, (a, b) in enumerate(zip(BAND_EDGES[:-1], BAND_EDGES[1:]))]
    values = []
    for c in range(2):
        values += [rms[c], stats.kurtosis(x[:, c], fisher=False, bias=False), np.max(np.abs(x[:, c])) / rms[c]]
        values += [float(spectra[c][m].sum() * df) for m in masks]
    product = pxx * pyy
    coh = np.divide(np.abs(pxy) ** 2, product, out=np.zeros_like(product), where=product > 1e-30)
    coh = np.clip(coh, 0, 1)
    phase = np.angle(pxy)  # scipy convention: conj(horizontal) * vertical
    for mask in masks:
        weight = np.sqrt(np.maximum(product[mask], 0))
        weight = weight / weight.sum() if weight.sum() > 1e-25 else np.zeros_like(weight)
        values += [np.sum(weight * coh[mask]), np.sum(weight * coh[mask] * np.sin(phase[mask])),
                   np.sum(weight * coh[mask] * np.cos(phase[mask]))]
    result = np.asarray(values, dtype=np.float64)
    if not np.isfinite(result).all(): raise ValueError('Invalid extracted features.')
    return result

def validate_features(values):
    if not isinstance(values, list) or len(values) != N_FEATURES:
        raise ValueError(f'Each capture needs {N_FEATURES} features in feature_names order.')
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in values):
        raise ValueError('Features must be numeric; booleans and numeric strings are not accepted.')
    x = np.asarray(values, dtype=float)
    if not np.isfinite(x).all() or (np.abs(x) > 1e15).any(): raise ValueError('Features must be finite and bounded.')
    if (x[LOG_INDICES] < 0).any(): raise ValueError('Energy and amplitude concepts cannot be negative.')
    if ((x[GROUPS['coherence']] < 0) | (x[GROUPS['coherence']] > 1)).any(): raise ValueError('Coherence must be in [0,1].')
    if (np.abs(x[PHASE_INDICES]) > 1 + 1e-8).any(): raise ValueError('Weighted sine/cosine features must be in [-1,1].')
    for band in range(8):
        i = 22 + 3 * band
        if np.hypot(x[i+1], x[i+2]) > x[i] + 1e-6:
            raise ValueError('Phase-vector magnitude cannot exceed band coherence.')
    return x

def concept_summary(x):
    x = np.asarray(x)
    return {'rms': float(np.mean(x[GROUPS['amplitude']])),
            'kurtosis': float(np.mean(x[[1, 12]])),
            'crest_factor': float(np.mean(x[[2, 13]])),
            'spectral_energy': float(np.sum(x[GROUPS['spectral_energy']])),
            'coherence': float(np.mean(x[GROUPS['coherence']])),
            'phase_consistency': float(np.mean([np.hypot(x[23+3*b], x[24+3*b]) for b in range(8)]))}
