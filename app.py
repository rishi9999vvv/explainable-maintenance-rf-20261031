"""Offline local dashboard. Run python app.py, then http://127.0.0.1:5055."""
from __future__ import annotations
import json, threading,zipfile,io
from pathlib import Path
from flask import Flask, Request, jsonify, render_template, request, send_file
from pdm.features import FEATURE_NAMES, BAND_EDGES
from pdm.inference_rf import Predictor
from pdm.upload import parse_upload,score_uploaded

ROOT=Path(__file__).parent

class MemoryUploadRequest(Request):
    def _get_file_stream(self,total_content_length,content_type,filename=None,content_length=None):
        return io.BytesIO()

def create_app(artifacts=None):
    artifact_root=Path(artifacts) if artifacts else ROOT/'artifacts_rf'
    app=Flask(__name__)
    app.request_class=MemoryUploadRequest
    app.config['MAX_CONTENT_LENGTH']=128*1024*1024
    predictor=Predictor(artifact_root)
    metrics=json.loads((artifact_root/'metrics.json').read_text(encoding='utf-8'))
    replay=json.loads((artifact_root/'replay.json').read_text(encoding='utf-8'))
    known_manifest=json.loads((ROOT/'data/xjtu/manifest.json').read_text(encoding='utf-8'))
    lock=threading.Lock()

    @app.after_request
    def headers(response):
        response.headers['X-Content-Type-Options']='nosniff'
        response.headers['Cache-Control']='no-store' if request.path.startswith('/api') else 'no-cache'
        return response

    @app.get('/')
    def index(): return render_template('index.html')

    @app.get('/api/health')
    def health(): return jsonify(status='ok',source_kind=predictor.config['source_kind'],primary_model='reference_rf',real_data_experiment='complete')

    @app.get('/api/v1/schema')
    def schema():
        return jsonify(schema_version=1,feature_names=FEATURE_NAMES,band_edges_hz=BAND_EDGES.tolist(),
                       window_captures=16,horizon_captures=10,capture_period_seconds=60,
                       supported_context=predictor.config['supported_context'],models=predictor.config['models'],
                       prediction_payload={'schema_version':1,'model':'reference_rf','bearing_id':'external-bearing',
                                           'operating_context':{'speed_rpm':2100,'load_kn':12},
                                           'captures':'16–256 entries: capture_index, elapsed_seconds, features (46 numeric values)'})

    @app.get('/api/v1/bearings')
    def bearings():
        return jsonify([{'bearing_id':name,'captures':len(rows),'partition':rows[0]['partition'],
                         'endpoint_index':rows[-1]['capture_index'],'speed_rpm':rows[0]['speed_rpm'],'load_kn':rows[0]['load_kn']}
                        for name,rows in replay.items()])

    @app.get('/api/v1/replay/<bearing>')
    def get_replay(bearing):
        if bearing not in replay: return jsonify(error='Unknown bearing.'),404
        return jsonify(source_kind=predictor.config['source_kind'],mode='offline labelled experiment replay',rows=replay[bearing])

    @app.get('/api/v1/metrics')
    def get_metrics(): return jsonify(metrics)

    @app.get('/api/v1/ai4i-results')
    def ai4i_results():
        return jsonify(json.loads((ROOT/'ai4i_baseline/results/metrics.json').read_text(encoding='utf-8')))

    @app.post('/api/v1/predict')
    def predict():
        payload=request.get_json(silent=True)
        try:
            with lock: result=predictor.predict(payload)
        except ValueError as exc: return jsonify(error=str(exc)),400
        return jsonify(result)

    @app.post('/api/v1/upload')
    def upload():
        try:
            for k,expected in [('speed_rpm',2100),('load_kn',12),('sample_rate_hz',25600),('capture_period_seconds',60)]:
                if float(request.form.get(k,'nan'))!=expected:raise ValueError('Use 2100 rpm, 12 kN, 25600 Hz and 60-second capture spacing.')
            bearing=request.form.get('bearing_id','Uploaded bearing').strip()
            if not 1<=len(bearing)<=80:raise ValueError('bearing_id must contain 1–80 characters.')
            if request.form.get('reference_acknowledged')!='true':raise ValueError('Confirm that the first 16 captures are the intended reference period.')
            with lock:
                rows,origin,digests,endpoint=parse_upload(request.files.getlist('files'),known_manifest)
                rows=score_uploaded(predictor,rows,bearing,endpoint)
            return jsonify(schema_version=1,bearing_id=bearing,source_kind=origin,mode='uploaded history replay; verified experiment endpoint' if endpoint else 'uploaded history replay; endpoint unknown',
                           endpoint_index=endpoint['endpoint_index'] if endpoint else None,endpoint=endpoint,rows=rows,files=digests,reference_assumption=predictor.config['reference_assumption'])
        except (ValueError,KeyError,TypeError,OverflowError,zipfile.BadZipFile,RuntimeError,NotImplementedError) as exc:return jsonify(error=str(exc)),400

    @app.get('/api/v1/sample-raw')
    def sample_raw():return send_file(ROOT/'samples/synthetic_raw_history.zip',as_attachment=True)

    @app.get('/api/v1/export/<model>')
    def export(model):
        if model not in predictor.models: return jsonify(error='Unknown model.'),404
        return send_file(artifact_root/f'{model}_test_predictions.csv',as_attachment=True)

    @app.errorhandler(413)
    def too_large(error): return jsonify(error='Request exceeds 128 MiB. Upload a smaller history or a ZIP.'),413

    return app

if __name__=='__main__':
    create_app().run(host='127.0.0.1',port=5055,debug=False,threaded=True)
