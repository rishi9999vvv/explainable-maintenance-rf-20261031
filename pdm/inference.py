from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import torch
from .data import transform
from .features import FEATURE_NAMES, GROUPS, validate_features, concept_summary
from .metrics import warning_states
from .network import TemporalModel, configure_cpu

class Predictor:
    def __init__(self, artifacts: Path):
        configure_cpu()
        self.config=json.loads((artifacts/'config.json').read_text(encoding='utf-8'))
        self.models={}
        for name in ['phase_tcn','no_phase_tcn']:
            spec=self.config['models'][name]
            model=TemporalModel(len(spec['kept_indices']))
            model.load_state_dict(torch.load(artifacts/f'{name}.pt',map_location='cpu',weights_only=True))
            self.models[name]=model.eval()

    def validate(self,payload):
        if not isinstance(payload,dict) or set(payload)!={'schema_version','model','bearing_id','operating_context','captures'}:
            raise ValueError('Provide schema_version, model, bearing_id, operating_context and captures only.')
        if type(payload['schema_version']) is not int or payload['schema_version']!=1: raise ValueError('schema_version must be 1.')
        if not isinstance(payload['model'],str) or payload['model'] not in self.models: raise ValueError('Unknown temporal model.')
        if not isinstance(payload['bearing_id'],str) or not 1<=len(payload['bearing_id'])<=80: raise ValueError('bearing_id must be a nonempty string up to 80 characters.')
        context=payload['operating_context']
        if not isinstance(context,dict) or set(context)!={'speed_rpm','load_kn'}: raise ValueError('operating_context needs speed_rpm and load_kn.')
        for k,expected in self.config['supported_context'].items():
            if type(context[k]) not in (int,float) or not np.isfinite(context[k]) or context[k]!=expected:
                raise ValueError('This model supports only 2100 rpm and 12 kN. Other conditions require new validation.')
        captures=payload['captures']
        if not isinstance(captures,list) or not 16<=len(captures)<=256: raise ValueError('Provide 16–256 chronological captures.')
        values=[]
        for i,c in enumerate(captures):
            if not isinstance(c,dict) or set(c)!={'capture_index','elapsed_seconds','features'}: raise ValueError('Each capture needs capture_index, elapsed_seconds, features.')
            if type(c['capture_index']) is not int or c['capture_index']<1: raise ValueError('capture_index must be a positive integer.')
            if type(c['elapsed_seconds']) not in (int,float) or not np.isfinite(c['elapsed_seconds']) or c['elapsed_seconds']<0:
                raise ValueError('elapsed_seconds must be finite and nonnegative.')
            if i and (c['capture_index']!=captures[i-1]['capture_index']+1 or abs(c['elapsed_seconds']-captures[i-1]['elapsed_seconds']-60)>1e-6):
                raise ValueError('Captures must be consecutive, aligned, and spaced 60 seconds apart. Gaps reset the required history.')
            values.append(validate_features(c['features']).tolist()+[context['speed_rpm'],context['load_kn']])
        return np.asarray(values),captures

    def scores(self,windows,name):
        kept=self.config['models'][name]['kept_indices']
        z=transform(windows,self.config['scaler'])[:,:,kept]
        with torch.no_grad(): return torch.sigmoid(self.models[name](torch.tensor(z))).numpy().astype(float)

    def predict(self,payload):
        begin=time.perf_counter()
        values,captures=self.validate(payload)
        name=payload['model']
        windows=np.stack([values[i-15:i+1] for i in range(15,len(values))])
        scores=self.scores(windows,name)
        meta=[{'bearing_id':payload['bearing_id'],'capture_index':r['capture_index'],'elapsed_seconds':r['elapsed_seconds']} for r in captures[15:]]
        threshold=self.config['models'][name]['threshold']
        active,starts=warning_states(scores,threshold,meta)
        reference=np.array(self.config['scaler']['reference'])
        explanations=[]
        latest=windows[-1:]
        for group,indices in GROUPS.items():
            if name=='no_phase_tcn' and group=='relative_phase': continue
            replaced=latest.copy()
            replaced[:,:,indices]=reference[indices]
            counterfactual=float(self.scores(replaced,name)[0])
            explanations.append({'group':group,'score_difference':float(scores[-1]-counterfactual),'reference_replaced_score':counterfactual})
        context_z=transform(latest,self.config['scaler'])
        feature_extreme=bool((np.abs(context_z[:,:,:len(FEATURE_NAMES)])>6).any())
        return {'schema_version':1,'source_kind':self.config['source_kind'],'model':name,
                'bearing_id':payload['bearing_id'],'score':float(scores[-1]),'threshold':threshold,
                'warning':bool(active[-1]),'warning_state':'warning' if active[-1] else 'pending_persistence' if scores[-1]>=threshold else 'below_threshold',
                'history_complete_for_persistence':len(scores)>=3,
                'score_type':'uncalibrated endpoint-warning score','horizon_captures':10,'horizon_minutes':10,
                'warning_history':[dict(**m,score=float(s),warning=bool(a),episode_start=bool(e)) for m,s,a,e in zip(meta,scores,active,starts)],
                'concepts':[{'capture_index':c['capture_index'],**concept_summary(c['features'])} for c in captures],
                'sensitivities':sorted(explanations,key=lambda r:abs(r['score_difference']),reverse=True),
                'sensitivity_note':'Replace one measured feature group throughout the last 16 captures with training medians. These correlated replacements are model sensitivities, not causal effects or additive SHAP explanations.',
                'quality_notes':(['Features exceed six training standard deviations; interpret with caution.'] if feature_extreme else [])+
                                (['Fewer than three model scores: persistent warning cannot yet be established.'] if len(scores)<3 else []),
                'elapsed_ms':(time.perf_counter()-begin)*1000}
