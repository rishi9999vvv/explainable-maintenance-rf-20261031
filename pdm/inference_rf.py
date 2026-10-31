import json,time
from pathlib import Path
import joblib,numpy as np
from .data import transform
from .features import FEATURE_NAMES,GROUPS,validate_features,concept_summary
from .forest import signatures,shap_explanation,SIGNAL_GROUPS,SIGNATURE_GROUPS,SIGNATURE_NAMES
from .metrics import warning_states
from .voting import causal_average

class Predictor:
    def __init__(self,artifacts:Path):
        self.config=json.loads((artifacts/'config.json').read_text())
        self.models={name:joblib.load(artifacts/f'{name}.joblib') for name in self.config['models']}
    def validate(self,p):
        if not isinstance(p,dict) or set(p)!={'schema_version','model','bearing_id','operating_context','captures'}:raise ValueError('Provide schema_version, model, bearing_id, operating_context and captures only.')
        if type(p['schema_version']) is not int or p['schema_version']!=1:raise ValueError('schema_version must be 1.')
        if not isinstance(p['model'],str) or p['model'] not in self.models:raise ValueError('Unknown Random Forest model.')
        if not isinstance(p['bearing_id'],str) or not 1<=len(p['bearing_id'])<=80:raise ValueError('Invalid bearing_id.')
        c=p['operating_context']
        if not isinstance(c,dict) or set(c)!={'speed_rpm','load_kn'}:raise ValueError('Provide speed_rpm and load_kn.')
        if any(type(c[k]) not in (int,float) or not np.isfinite(c[k]) or c[k]!=v for k,v in self.config['supported_context'].items()):raise ValueError('Supported context is 2100 rpm and 12 kN only.')
        caps=p['captures']
        if not isinstance(caps,list) or not 16<=len(caps)<=256:raise ValueError('Provide 16–256 captures, starting with the reference period.')
        values=[]
        for i,r in enumerate(caps):
            if not isinstance(r,dict) or set(r)!={'capture_index','elapsed_seconds','features'}:raise ValueError('Each capture needs capture_index, elapsed_seconds and features.')
            if type(r['capture_index']) is not int or r['capture_index']!=i+1:raise ValueError('History must start at capture 1 and contain no gaps or duplicates.')
            if type(r['elapsed_seconds']) not in (int,float) or not np.isfinite(r['elapsed_seconds']) or r['elapsed_seconds']!=i*60:raise ValueError('Captures must have 60-second spacing, starting at elapsed_seconds=0.')
            values.append(validate_features(r['features']).tolist()+[c['speed_rpm'],c['load_kn']])
        return np.array(values),caps
    def vectors(self,values,name):
        if name=='snapshot_rf':return transform(values[15:],self.config['scaler'])
        return signatures(values,np.array(self.config['reference_floor']))
    def score_values(self,values,name):
        x=self.vectors(values,name);kept=self.config['models'][name]['kept_indices']
        raw=self.models[name].predict_proba(x[:,kept])[:,1]
        history=self.config['models'][name].get('vote_history_captures',1)
        return causal_average(raw,history),x
    def predict(self,p):
        start=time.perf_counter();values,caps=self.validate(p);name=p['model'];scores,x=self.score_values(values,name)
        meta=[{'bearing_id':p['bearing_id'],'capture_index':r['capture_index'],'elapsed_seconds':r['elapsed_seconds']} for r in caps[15:]]
        threshold=self.config['models'][name]['threshold'];active,episodes=warning_states(scores,threshold,meta)
        kept=self.config['models'][name]['kept_indices']
        vote_count=min(len(x),self.config['models'][name].get('vote_history_captures',1))
        if name=='snapshot_rf':
            import shap
            explainer=shap.TreeExplainer(self.models[name]);sv=np.asarray(explainer.shap_values(x[-1:]))[0,:,1];base=float(explainer.expected_value[1]);groups={**GROUPS,'context':[46,47]};feature_names=FEATURE_NAMES+['speed_rpm','load_kn'];families={}
        else:
            all_sv,base=shap_explanation(self.models[name],x[-vote_count:],kept);sv=all_sv.mean(axis=0);groups=SIGNAL_GROUPS;feature_names=SIGNATURE_NAMES;families={k:float(sv[idx].sum()) for k,idx in SIGNATURE_GROUPS.items()}
        reconstruction=abs(base+sv.sum()-scores[-1])
        if reconstruction>1e-6:raise ValueError('SHAP reconstruction check failed.')
        contributions=[{'group':g,'score_difference':float(sv[idx].sum())} for g,idx in groups.items() if any(i in kept for i in idx)]
        top=sorted([{'feature':n,'shap':float(v),'value':float(z)} for n,v,z in zip(feature_names,sv,x[-vote_count:].mean(axis=0))],key=lambda r:abs(r['shap']),reverse=True)[:8]
        return {'schema_version':1,'source_kind':self.config['source_kind'],'model':name,'bearing_id':p['bearing_id'],'score':float(scores[-1]),'threshold':threshold,
            'warning':bool(active[-1]),'warning_state':'warning' if active[-1] else 'pending_persistence' if scores[-1]>=threshold else 'below_threshold',
            'history_complete_for_persistence':len(scores)>=3,'score_type':'uncalibrated endpoint-warning score','horizon_captures':10,'horizon_minutes':10,
            'warning_history':[dict(**m,score=float(s),warning=bool(a),episode_start=bool(e)) for m,s,a,e in zip(meta,scores,active,episodes)],
            'concepts':[dict(capture_index=r['capture_index'],**concept_summary(r['features'])) for r in caps],
            'sensitivities':sorted(contributions,key=lambda r:abs(r['score_difference']),reverse=True),'explanation_method':'TreeSHAP','shap_base':base,'shap_reconstruction_error':float(reconstruction),
            'signature_contributions':families,'top_features':top,'vote_history_captures':self.config['models'][name].get('vote_history_captures',1),
            'effective_vote_history_captures':vote_count,'explanation_aggregation':'Mean TreeSHAP contributions over the same past scoring windows as the displayed vote average.',
            'top_feature_value_representation':'Mean signature value over contributing scoring windows; this is not the forest prediction at mean inputs.',
            'sensitivity_note':'Grouped TreeSHAP contributions reconstruct the displayed vote-averaged score. Correlated inputs can share attribution; this is not physical causation.',
            'quality_notes':([self.config['reference_assumption']] if name!='snapshot_rf' else [])+(['Persistent warning needs at least 18 captures from a cold start.'] if len(scores)<3 else []),
            'elapsed_ms':(time.perf_counter()-start)*1000}
