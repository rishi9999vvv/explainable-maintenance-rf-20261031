"""Reproduce all model results. Run: python train.py. No test-driven tuning."""
import core
from core import *
import hashlib, json, platform, importlib.metadata, time
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.metrics import (average_precision_score, roc_auc_score, precision_score, recall_score,
    f1_score, fbeta_score, balanced_accuracy_score, accuracy_score, confusion_matrix,
    brier_score_loss, precision_recall_curve, roc_curve)

SEED = 42
OUT = ROOT / 'results'
FIG = OUT / 'figures'
for path in (OUT, FIG, ROOT / 'models'):
    path.mkdir(parents=True, exist_ok=True)

def metrics(y, p, threshold):
    pred = p >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(ap=float(average_precision_score(y,p)), roc_auc=float(roc_auc_score(y,p)),
        precision=float(precision_score(y,pred,zero_division=0)), recall=float(recall_score(y,pred,zero_division=0)),
        f1=float(f1_score(y,pred,zero_division=0)), f2=float(fbeta_score(y,pred,beta=2,zero_division=0)),
        accuracy=float(accuracy_score(y,pred)), balanced_accuracy=float(balanced_accuracy_score(y,pred)),
        brier=float(brier_score_loss(y,p)), tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp),threshold=float(threshold))

def choose_threshold(y, p):
    # Highest threshold breaks F2 ties, reducing unnecessary alerts.
    thresholds = np.unique(np.r_[0.0,p,1.0])
    precision, recall, th = precision_recall_curve(y,p)
    f2 = 5*precision[:-1]*recall[:-1]/np.maximum(4*precision[:-1]+recall[:-1],1e-15)
    best = np.max(f2)
    return float(np.max(th[np.isclose(f2,best)]))

def bootstrap(y,p,t,n=500):
    rng = np.random.default_rng(SEED)
    y,p=np.asarray(y),np.asarray(p)
    groups=[np.flatnonzero(y==k) for k in (0,1)]
    rows=[]
    for _ in range(n):
        idx=np.concatenate([rng.choice(g,len(g),replace=True) for g in groups])
        rows.append(metrics(y[idx],p[idx],t))
    return {k:np.percentile([r[k] for r in rows],[2.5,97.5]).tolist() for k in ['ap','roc_auc','precision','recall','f1','f2']}

def main():
    start=time.time()
    raw_path=ROOT/'data/ai4i2020.csv'
    df=pd.read_csv(raw_path)
    assert len(df)==10000 and set(FEATURES+FORBIDDEN)<=set(df.columns)
    assert not df.isna().any().any()
    assert set(df[TARGET])=={0,1} and set(df.Type)=={'L','M','H'}
    X,y=df[FEATURES],df[TARGET]
    trainval,test=train_test_split(np.arange(len(df)),test_size=.2,stratify=y,random_state=SEED)
    train,val=train_test_split(trainval,test_size=.25,stratify=y.iloc[trainval],random_state=SEED)
    assert not (set(train)&set(val) or set(train)&set(test) or set(val)&set(test))
    hashes=pd.util.hash_pandas_object(X,index=False)
    overlap={f'{a}_{b}':len(set(hashes.iloc[i])&set(hashes.iloc[j])) for a,i,b,j in [('train',train,'val',val),('train',train,'test',test),('val',val,'test',test)]}
    assert all(v==0 for v in overlap.values()),'Duplicate feature rows across splits; use group split'
    audit=dict(rows=len(df),failures=int(y.sum()),prevalence=float(y.mean()),missing=int(df.isna().sum().sum()),
        feature_duplicates=int(X.duplicated().sum()),split_overlap=overlap,
        excluded=FORBIDDEN,features=FEATURES,sha256=hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        flag_target_disagreements=int((df[['TWF','HDF','PWF','OSF','RNF']].max(axis=1)!=y).sum()),
        splits={k:{'n':len(i),'failures':int(y.iloc[i].sum())} for k,i in [('train',train),('validation',val),('test',test)]})
    split=pd.DataFrame({'UDI':df.UDI,'partition':''})
    for name,idx in [('train',train),('validation',val),('test',test)]: split.loc[idx,'partition']=name
    split.to_csv(OUT/'split_manifest.csv',index=False)
    cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=SEED)
    baseline={
        'Dummy':make_pipeline(DummyClassifier(strategy='prior')),
        'Logistic':make_pipeline(LogisticRegression(class_weight='balanced',max_iter=2000,random_state=SEED),scale=True),
        'Raw RF':make_pipeline(RandomForestClassifier(n_estimators=250,min_samples_leaf=2,class_weight='balanced_subsample',random_state=SEED,n_jobs=2))}
    cv_rows=[]
    for name,pipe in baseline.items():
        print('Cross-validating',name,flush=True)
        scores=cross_val_score(pipe,X.iloc[train],y.iloc[train],cv=cv,scoring='average_precision',n_jobs=1)
        cv_rows.append(dict(model=name,mean_ap=float(scores.mean()),std_ap=float(scores.std()),fold_ap=scores.tolist()))
        pipe.fit(X.iloc[train],y.iloc[train])
    improved=make_pipeline(RandomForestClassifier(n_estimators=250,class_weight='balanced_subsample',random_state=SEED,n_jobs=2),engineered=True)
    search=GridSearchCV(improved,{'model__max_depth':[8,None],'model__min_samples_leaf':[1,3]},scoring='average_precision',cv=cv,n_jobs=1,refit=True)
    print('Tuning engineered RF on training folds only',flush=True)
    search.fit(X.iloc[train],y.iloc[train])
    baseline['Engineered RF']=search.best_estimator_
    cv_rows.append(dict(model='Engineered RF',mean_ap=float(search.best_score_),std_ap=float(search.cv_results_['std_test_score'][search.best_index_]),fold_ap=[float(search.cv_results_[f'split{k}_test_score'][search.best_index_]) for k in range(5)]))
    pd.DataFrame(search.cv_results_).drop(columns=['params']).to_csv(OUT/'cv_search.csv',index=False)
    # Predeclared primary model: tuned engineered random forest. Test never selects the model.
    records=[]; predictions={}; thresholds={}
    for name,pipe in baseline.items():
        pv=pipe.predict_proba(X.iloc[val])[:,1]
        threshold=1.0 if name=='Dummy' else choose_threshold(y.iloc[val],pv)
        thresholds[name]=threshold
        p=pipe.predict_proba(X.iloc[test])[:,1]
        predictions[name]=p
        records.append(dict(model=name,validation=metrics(y.iloc[val],pv,threshold),test=metrics(y.iloc[test],p,threshold),test_at_05=metrics(y.iloc[test],p,.5)))
    model=baseline['Engineered RF']; threshold=thresholds['Engineered RF']; p=predictions['Engineered RF']
    ci=bootstrap(y.iloc[test],p,threshold)
    joblib.dump({'pipeline':model,'threshold':threshold,'features':FEATURES,'sha256':audit['sha256']},ROOT/'models/model.joblib')
    test_frame=df.iloc[test][['UDI','Type',TARGET]].copy()
    for name,pred in predictions.items(): test_frame[name+' score']=pred
    test_frame.to_csv(OUT/'test_predictions.csv',index=False)
    # Independent ordered sensitivity: fixed parameters declared here, never selected using random-test results.
    ordered=make_pipeline(RandomForestClassifier(n_estimators=250,min_samples_leaf=3,max_depth=8,class_weight='balanced_subsample',random_state=SEED,n_jobs=2),engineered=True)
    ordered.fit(X.iloc[:6000],y.iloc[:6000])
    bt=choose_threshold(y.iloc[6000:8000],ordered.predict_proba(X.iloc[6000:8000])[:,1])
    blocked=metrics(y.iloc[8000:],ordered.predict_proba(X.iloc[8000:])[:,1],bt)
    blocked['split_failures']=[int(y.iloc[a:b].sum()) for a,b in [(0,6000),(6000,8000),(8000,10000)]]
    subgroups={str(g):dict(n=int((df.iloc[test].Type==g).sum()),**metrics(y.iloc[test][df.iloc[test].Type==g],p[(df.iloc[test].Type==g).to_numpy()],threshold)) for g in ['L','M','H']}
    print('Computing SHAP and confidence intervals',flush=True)
    rng=np.random.default_rng(SEED)
    sample=rng.choice(test,size=200,replace=False)
    values,base,names=explain(model,X.iloc[sample])
    importance=np.abs(values).mean(axis=0)
    pd.DataFrame({'feature':names,'mean_abs_shap':importance}).sort_values('mean_abs_shap',ascending=False).to_csv(OUT/'shap_importance.csv',index=False)
    np.savez_compressed(OUT/'shap_values.npz',values=values,base=base,udi=df.iloc[sample].UDI.to_numpy(),names=names)
    examples={}
    for label,mask in [('true_positive',(y.iloc[test].to_numpy()==1)&(p>=threshold)),('true_negative',(y.iloc[test].to_numpy()==0)&(p<threshold)),('false_negative',(y.iloc[test].to_numpy()==1)&(p<threshold)),('false_positive',(y.iloc[test].to_numpy()==0)&(p>=threshold))]:
        positions=np.flatnonzero(mask)
        if len(positions):
            pos=int(positions[0]); idx=int(test[pos]); v,b,ns=explain(model,X.iloc[[idx]])
            examples[label]=dict(udi=int(df.iloc[idx].UDI),inputs=X.iloc[idx].to_dict(),actual=int(y.iloc[idx]),score=float(p[pos]),base=b,shap=dict(zip(ns,v[0].tolist())))
    (OUT/'demo_examples.json').write_text(json.dumps(examples,indent=2))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(8,5))
    for name,scores in predictions.items():
        pr,re,_=precision_recall_curve(y.iloc[test],scores)
        ax.step(re,pr,where='post',label=f'{name} (AP {average_precision_score(y.iloc[test],scores):.3f})',lw=2)
    ax.axhline(y.iloc[test].mean(),color='gray',ls='--',lw=1)
    ax.set(xlabel='Recall',ylabel='Precision',title='Held-out precision-recall curves',xlim=(0,1),ylim=(0,1.02));ax.legend(fontsize=9);fig.tight_layout();fig.savefig(FIG/'precision_recall.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4.5)); order=np.argsort(importance)
    ax.barh(np.array(names)[order],importance[order],color='#087f8c');ax.set(xlabel='Mean absolute SHAP contribution to failure score',title='Global SHAP: 200 randomly selected test readings');fig.tight_layout();fig.savefig(FIG/'shap_global.png',dpi=160);plt.close(fig)
    result=dict(seed=SEED,audit=audit,cv=cv_rows,best_params=search.best_params_,models=records,primary='Engineered RF',confidence_intervals=ci,ordered_holdout=blocked,subgroups=subgroups,shap_base=base,shap_sample_n=200,
        methodology='60/20/20 stratified; 5-fold train-only AP selection; validation F2 threshold; no refit after threshold selection',
        versions={m:importlib.metadata.version(m) for m in ['numpy','pandas','scikit-learn','scipy','shap','joblib','matplotlib','flask']},python=platform.python_version(),elapsed_seconds=time.time()-start)
    (OUT/'metrics.json').write_text(json.dumps(result,indent=2))
    pd.DataFrame([dict(model=r['model'],**r['test']) for r in records]).to_csv(OUT/'metrics.csv',index=False)
    (ROOT/'requirements.txt').write_text('\n'.join(f'{k}=={v}' for k,v in result['versions'].items())+'\n')
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__': main()
