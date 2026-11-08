from src.model import *
from src.clean_data import *
from src.model_config import *
from src.preprocessing import *
from sklearn.metrics import precision_recall_curve

def train(model_params, param_grid):
    x, y = split_data(read("train"))

    pip_lin = pipeline(model_params)
    grd_srch = grid_search(pip_lin, param_grid)
    print("prepared data")
    
    grd_srch.fit(x, y)
    model_params["alpha"] = grd_srch.best_params_["model__alpha"]
    print("trained model")
    
    return save_model(model_params, grd_srch.best_estimator_)

def calc_threshold(model_name, data):
    x, y = split_data(data)
    model = open_model(model_name)
    score = model.predict_proba(x)[:, 1]
    
    precisions, recalls, thresholds = precision_recall_curve(y, score)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    threshold = thresholds[np.argmax(f1_scores[:-1])]

    model_params = get_hyperparameters(model_name)
    model_params["threshold"] = float(threshold)

    save_model(model_params, name=model_name)
    print("calc best threshold")
    return threshold