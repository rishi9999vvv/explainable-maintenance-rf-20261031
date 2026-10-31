"""Local demonstration. Run python app.py and open http://127.0.0.1:5000."""
import core
from core import *
import json, math, joblib
from flask import Flask, jsonify, request, render_template, send_from_directory

app=Flask(__name__)
app.config['MAX_CONTENT_LENGTH']=16*1024
bundle=joblib.load(ROOT/'models/model.joblib')
model,threshold=bundle['pipeline'],bundle['threshold']
data=pd.read_csv(ROOT/'data/ai4i2020.csv')
manifest=pd.read_csv(ROOT/'results/split_manifest.csv')
train_data=data[data.UDI.isin(manifest.loc[manifest.partition=='train','UDI'])]
ranges={k:[float(train_data[k].min()),float(train_data[k].max())] for k in NUMERIC}
examples=json.loads((ROOT/'results/demo_examples.json').read_text())

@app.get('/')
def index():
    return render_template('index.html',fields=NUMERIC,ranges=ranges,threshold=threshold)

@app.get('/api/examples')
def get_examples():
    return jsonify(examples)

@app.get('/deliverables/<path:filename>')
def deliverable(filename):
    return send_from_directory(ROOT/'deliverables',filename)

@app.get('/results/figures/<path:filename>')
def figure(filename):
    return send_from_directory(ROOT/'results/figures',filename)

@app.get('/api/health')
def health():
    return jsonify(status='ok',model='Engineered RF',threshold=threshold)

@app.post('/api/predict')
def predict():
    payload=request.get_json(silent=True)
    if not isinstance(payload,dict) or set(payload)!=set(FEATURES):
        return jsonify(error='Provide exactly Type and the five sensor fields.'),400
    if payload['Type'] not in ['L','M','H']:
        return jsonify(error='Type must be L, M, or H.'),400
    clean={'Type':payload['Type']}
    for name in NUMERIC:
        try:
            if isinstance(payload[name],bool): raise ValueError()
            value=float(payload[name])
            if not math.isfinite(value) or not ranges[name][0]<=value<=ranges[name][1]: raise ValueError()
            clean[name]=value
        except (ValueError,TypeError):
            return jsonify(error=f'{name} must be within the training range {ranges[name]}. This demo does not extrapolate.'),400
    X=pd.DataFrame([clean],columns=FEATURES)
    score=float(model.predict_proba(X)[0,1])
    values,base,names=explain(model,X)
    contributions=sorted([dict(feature=k,value=float(v)) for k,v in zip(names,values[0])],key=lambda r:abs(r['value']),reverse=True)
    return jsonify(score=score,threshold=threshold,alert=score>=threshold,base=base,contributions=contributions,
        reconstruction_error=abs(base+values[0].sum()-score))

if __name__=='__main__':
    app.run(host='127.0.0.1',port=5000,debug=False)
