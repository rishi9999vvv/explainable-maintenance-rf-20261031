from __future__ import annotations
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, precision_score, recall_score, fbeta_score

def choose_threshold(y, scores):
    precision, recall, thresholds = precision_recall_curve(y, scores)
    f2 = 5 * precision[:-1] * recall[:-1] / np.maximum(4 * precision[:-1] + recall[:-1], 1e-15)
    return float(np.max(thresholds[np.isclose(f2, np.max(f2))]))

def warning_states(scores, threshold, metadata, persistence=3):
    states, starts = [], []
    streak, previous, active = 0, None, False
    for score, row in zip(scores, metadata):
        contiguous = previous is not None and row['bearing_id'] == previous['bearing_id'] and row['capture_index'] == previous['capture_index'] + 1 and row['elapsed_seconds'] == previous['elapsed_seconds'] + 60
        if not contiguous: streak, active = 0, False
        streak = streak + 1 if score >= threshold else 0
        now = streak >= persistence
        starts.append(now and not active)
        states.append(now)
        active, previous = now, row
    return np.array(states, dtype=bool), np.array(starts, dtype=bool)

def evaluate(y, scores, threshold, metadata):
    y, scores = np.asarray(y), np.asarray(scores)
    pred = scores >= threshold
    active, starts = warning_states(scores, threshold, metadata)
    episodes = []
    endpoint_rows = []
    for i in np.flatnonzero(starts):
        lead = metadata[i]['endpoint_index'] - metadata[i]['capture_index']
        episodes.append({**metadata[i], 'lead_captures': lead, 'outside_target_horizon': not (0 < lead <= 10)})
    for bearing in sorted(set(r['bearing_id'] for r in metadata)):
        indices = [i for i,r in enumerate(metadata) if r['bearing_id'] == bearing]
        alerts = [i for i in indices if active[i]]
        timely = [i for i in alerts if y[i] == 1]
        endpoint_rows.append({'bearing_id': bearing,
                              'first_warning_lead_captures': metadata[alerts[0]]['endpoint_index'] - metadata[alerts[0]]['capture_index'] if alerts else None,
                              'first_in_horizon_warning_lead_captures': metadata[timely[0]]['endpoint_index'] - metadata[timely[0]]['capture_index'] if timely else None,
                              'missed_endpoint': not bool(timely)})
    return {'average_precision': float(average_precision_score(y, scores)),
            'precision': float(precision_score(y,pred,zero_division=0)),
            'recall': float(recall_score(y,pred,zero_division=0)),
            'f2': float(fbeta_score(y,pred,beta=2,zero_division=0)),
            'threshold': float(threshold), 'n_windows': len(y), 'positive_windows': int(y.sum()),
            'tp': int(np.sum(pred & (y==1))), 'fp': int(np.sum(pred & (y==0))),
            'fn': int(np.sum(~pred & (y==1))), 'tn': int(np.sum(~pred & (y==0))),
            'persistent_warning_recall': float(np.sum(active & (y==1))/max(1,y.sum())),
            'warning_episodes': episodes,
            'episodes_outside_target_horizon': sum(r['outside_target_horizon'] for r in episodes),
            'outside_horizon_episodes_per_evaluated_hour': sum(r['outside_target_horizon'] for r in episodes)/(len(y)/60),
            'evaluated_capture_hours': len(y)/60, 'endpoints': endpoint_rows,
            'missed_endpoints': sum(r['missed_endpoint'] for r in endpoint_rows)}
