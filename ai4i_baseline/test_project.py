"""Integration tests of saved artifacts, model isolation, and demonstration API."""
import unittest, json
import core
from core import *
import joblib

class ProjectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from app import app,examples
        cls.client=app.test_client()
        cls.examples=examples
        cls.bundle=joblib.load(ROOT/'models/model.joblib')
    def test_valid_prediction_and_shap(self):
        for ex in self.examples.values():
            response=self.client.post('/api/predict',json=ex['inputs'])
            self.assertEqual(response.status_code,200,response.json)
            self.assertAlmostEqual(response.json['score'],ex['score'],places=12)
            self.assertLess(response.json['reconstruction_error'],1e-6)
            self.assertEqual(response.json['alert'],ex['score']>=self.bundle['threshold'])
    def test_reject_leakage_fields(self):
        ex=dict(next(iter(self.examples.values()))['inputs']);ex['Machine failure']=1
        self.assertEqual(self.client.post('/api/predict',json=ex).status_code,400)
    def test_invalid_input(self):
        for field,value in [('Type','Unknown'),(NUMERIC[0],None),(NUMERIC[1],float('nan')),(NUMERIC[2],-1),(NUMERIC[3],True)]:
            ex=dict(next(iter(self.examples.values()))['inputs']);ex[field]=value
            self.assertEqual(self.client.post('/api/predict',json=ex).status_code,400)
        self.assertEqual(self.client.post('/api/predict',json={}).status_code,400)
    def test_pipeline_ignores_outcomes(self):
        x=pd.DataFrame([next(iter(self.examples.values()))['inputs']]); altered=x.copy()
        for col in FORBIDDEN: altered[col]=999
        np.testing.assert_allclose(self.bundle['pipeline'].predict_proba(x),self.bundle['pipeline'].predict_proba(altered))
    def test_splits_and_metrics(self):
        m=json.loads((ROOT/'results/metrics.json').read_text())
        self.assertTrue(all(v==0 for v in m['audit']['split_overlap'].values()))
        for row in m['models']:
            t=row['test'];self.assertEqual(t['tn']+t['fp']+t['fn']+t['tp'],2000)
        manifest=pd.read_csv(ROOT/'results/split_manifest.csv')
        self.assertEqual(manifest.UDI.nunique(),10000)
    def test_page(self):
        self.assertEqual(self.client.get('/').status_code,200)
        self.assertEqual(self.client.get('/api/health').json['status'],'ok')
        for path in ['/deliverables/PROJECT_REPORT.html','/results/figures/shap_global.png']:
            with self.client.get(path) as response:
                self.assertEqual(response.status_code,200)

if __name__=='__main__': unittest.main(verbosity=2)
