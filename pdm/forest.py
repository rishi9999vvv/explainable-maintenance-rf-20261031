"""Self-referenced temporal signatures for an explainable Random Forest."""
import numpy as np
from .data import compress
from .features import FEATURE_NAMES, GROUPS

SIGNATURE_NAMES=[f'{kind}__{name}' for kind in ['reference_deviation','recent_trend','local_variability'] for name in FEATURE_NAMES]+['speed_rpm','load_kn']
SIGNAL_GROUPS={name:[offset+i for offset in [0,46,92] for i in indices] for name,indices in GROUPS.items()}
SIGNAL_GROUPS['context']=[138,139]
SIGNATURE_GROUPS={'reference_deviation':list(range(46)),'recent_trend':list(range(46,92)),'local_variability':list(range(92,138)),'context':[138,139]}

def fit_reference_floor(dataset,train_ids):
    scales=[]
    for bearing in train_ids:
        rows=sorted([r for r in dataset['rows'] if r['bearing_id']==bearing],key=lambda r:r['capture_index'])[:16]
        x=compress(np.array([r['features'] for r in rows]))
        med=np.median(x,axis=0)
        scales.append(1.4826*np.median(np.abs(x-med),axis=0))
    return np.maximum(np.median(scales,axis=0)*.25,.02)

def reference(captures,floor):
    if len(captures)<16:raise ValueError('Reference requires the first 16 chronological captures.')
    x=compress(np.asarray(captures)[:16,:46])
    med=np.median(x,axis=0)
    scale=np.maximum(1.4826*np.median(np.abs(x-med),axis=0),floor)
    return med,scale

def signatures(captures,floor):
    raw=np.asarray(captures,dtype=float)
    med,scale=reference(raw,floor)
    x=compress(raw[:,:46])
    result=[]
    for end in range(15,len(x)):
        window=x[end-15:end+1]
        result.append(np.r_[(x[end]-med)/scale,(window[-4:].mean(axis=0)-window[:4].mean(axis=0))/scale,
                            window[-4:].std(axis=0)/scale,raw[end,46:48]])
    return np.asarray(result,dtype=np.float32)

def dataset_signatures(data,ids,floor):
    from .data import HORIZON
    xs=[];ys=[];meta=[]
    for bearing in ids:
        rows=sorted([r for r in data['rows'] if r['bearing_id']==bearing],key=lambda r:r['capture_index'])
        if [r['capture_index'] for r in rows]!=list(range(1,len(rows)+1)):raise ValueError('Training histories must begin at capture 1 and be complete.')
        values=[r['features']+[r['speed_rpm'],r['load_kn']] for r in rows]
        z=signatures(values,floor)
        for j,row in enumerate(rows[15:-1]):
            remaining=rows[-1]['capture_index']-row['capture_index']
            xs.append(z[j]);ys.append(int(0<remaining<=HORIZON));meta.append({'bearing_id':bearing,'capture_index':row['capture_index'],
                'elapsed_seconds':row['elapsed_seconds'],'endpoint_index':rows[-1]['capture_index']})
    return np.stack(xs),np.asarray(ys),meta

def shap_explanation(model,x,kept):
    import shap
    x=np.asarray(x)
    explainer=shap.TreeExplainer(model)
    sv=np.asarray(explainer.shap_values(x[:,kept]))[:,:,1]
    base=float(explainer.expected_value[1])
    score=model.predict_proba(x[:,kept])[:,1]
    if not np.allclose(base+sv.sum(axis=1),score,atol=1e-6):raise ValueError('SHAP reconstruction failed.')
    full=np.zeros((len(x),140));full[:,kept]=sv
    return full,base
