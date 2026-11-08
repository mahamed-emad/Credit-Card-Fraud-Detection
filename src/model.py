import numpy as np
import pandas as pd
from .functions import *
from .clean_data import *
from .model_config import *
from .preprocessing import *
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.preprocessing import PolynomialFeatures, FunctionTransformer

def pipeline(model_params):
    preprocess = ColumnTransformer([
        ("hour", OneHotEncoder(handle_unknown="ignore"), ["Time"]),
        ("amt", Pipeline([("log", FunctionTransformer(np.log1p)),
                    ("sc", StandardScaler())]), ["Amount"])
    ], remainder='passthrough')

    _pipeline = Pipeline(steps=[
        # ("cln_clms", FunctionTransformer(clean_clms)),
        # ("to_np", FunctionTransformer(to_np)),
        ("add_clms", FunctionTransformer(add_clms)),
        # ("delete", FunctionTransformer(delete)),
        ("clms_trans", preprocess),
        ("sampler", RandomUnderSampler(sampling_strategy=0.1, random_state=42)),
        ("model", MLPClassifier(**filter_params(MLPClassifier, model_params)))
    ])
    return _pipeline

def grid_search(pipeline, param_grid):
    _grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=4,
        scoring='average_precision',
        n_jobs=3
    )
    return _grid_search