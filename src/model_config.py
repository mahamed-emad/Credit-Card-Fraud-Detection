from pathlib import Path
import json, joblib, inspect

def read_json():
    path = "models/experiments.json"
    json_file = Path(path)
    if json_file.exists():
        with open(json_file, "r", encoding="utf-8") as f:
            try:
                experiments = json.load(f)
                if not isinstance(experiments, dict):
                    experiments = {}
            except json.JSONDecodeError:
                experiments = {}
    else: experiments = {}
    return experiments

def save_model(hyperparameters, model=None, name="model"):
    path, extn = "models/", ".pkl"
    experiments = read_json()
    if name == "model":
        num = str(len(experiments))
        while name + num in experiments:
            num = str(int(num) + 1)
        name += num
    experiments[name] = hyperparameters
    with open("models/experiments.json", "w", encoding="utf-8") as f:
        json.dump(experiments, f, ensure_ascii=False, indent=4)
    if model: joblib.dump(model, path + name + extn)
    return name

def get_hyperparameters(name="model0"):
    json_file = Path("models/experiments.json")
    if not json_file.exists(): return {}
    with open(json_file, "r", encoding="utf-8") as f:
        experiments = json.load(f)
    if name in experiments: return experiments[name]

def open_model(name="model0"):
    path, extn = "models/", ".pkl"
    return joblib.load(path + name + extn)

def filter_params(func, params):
    sig = inspect.signature(func)
    allowed_keys = sig.parameters.keys()
    filtered = {k: params[k] for k in allowed_keys if k in params}
    return filtered