"""Shared feature construction and inference. Only pre-outcome sensor inputs are allowed."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent
if (ROOT / '.packages').exists():
    sys.path.insert(0, str(ROOT / '.packages'))
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

NUMERIC = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
FEATURES = ['Type'] + NUMERIC
TARGET = 'Machine failure'
FORBIDDEN = ['UDI', 'Product ID', TARGET, 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
DERIVED = ['Temperature difference [K]', 'Mechanical power [W]', 'Wear x torque [min Nm]']

class EngineeringFeatures(TransformerMixin, BaseEstimator):
    def __init__(self, enabled=True):
        self.enabled = enabled
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X[FEATURES].copy()
        if self.enabled:
            X[DERIVED[0]] = X[NUMERIC[1]] - X[NUMERIC[0]]
            X[DERIVED[1]] = X[NUMERIC[3]] * X[NUMERIC[2]] * 2 * np.pi / 60
            X[DERIVED[2]] = X[NUMERIC[4]] * X[NUMERIC[3]]
        return X

def make_pipeline(model, engineered=False, scale=False):
    nums = NUMERIC + (DERIVED if engineered else [])
    prep = ColumnTransformer([
        ('numeric', StandardScaler() if scale else 'passthrough', nums),
        ('type', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['Type'])
    ], verbose_feature_names_out=False)
    return Pipeline([('engineering', EngineeringFeatures(engineered)), ('preprocess', prep), ('model', model)])

def transformed(pipe, X):
    return pipe.named_steps['preprocess'].transform(pipe.named_steps['engineering'].transform(X))

def explain(pipe, X):
    import shap
    Z = transformed(pipe, X)
    explainer = shap.TreeExplainer(pipe.named_steps['model'], feature_perturbation='tree_path_dependent', model_output='raw')
    values = np.asarray(explainer.shap_values(Z))[:, :, 1]
    base = float(explainer.expected_value[1])
    scores = pipe.predict_proba(X)[:, 1]
    assert np.allclose(base + values.sum(axis=1), scores, atol=1e-6), 'SHAP additivity failed'
    return values, base, pipe.named_steps['preprocess'].get_feature_names_out().tolist()
