import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score

def to_np(data):
    return data.to_numpy()

def to_pd(data):
    return pd.DataFrame(data)

def log(clm):
    return np.log1p(clm)

def exp(clm):
    return np.expm1(clm)

def split_data(data, targets=["Class"]):
    clms_x = [clm for clm in data.columns if clm not in targets]
    return data[clms_x], data[targets].to_numpy().ravel() if targets[0] in data.columns else None

def normalize(x, mn, mx):
    x = (x - mn) / ((mx - mn) + 1e-9)
    return x

def calc_normalize(x):
    mn, mx = np.min(x, axis=0), np.max(x, axis=0)
    return mn, mx

def calc_error(score, y_true, model_params, name="val"):
    ap_val = average_precision_score(y_true, score)
    threshold = model_params["threshold"]
    pred = (score >= threshold).astype(int)
    f1_val = f1_score(y_true, pred)
    model_params[f"ap_{name}"] = ap_val
    model_params[f"f1_{name}"] = f1_val
    return model_params