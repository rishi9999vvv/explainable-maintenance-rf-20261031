"""Past-only score averaging with deterministic decision precision."""
import numpy as np

def causal_average(scores,history):
    if type(history) is not int or history<1:raise ValueError('Vote history must be a positive integer.')
    a=np.asarray(scores,dtype=float)
    if a.ndim!=1 or not np.isfinite(a).all():raise ValueError('Expected a finite one-dimensional score history.')
    if history==1:return a.copy()
    # Direct short-window means avoid cumulative-sum cancellation. Decision scores
    # are rounded to 12 decimals so threshold ties cannot exploit floating noise.
    return np.round([np.mean(a[max(0,i-history+1):i+1]) for i in range(len(a))],12)
