from src import *

def test(model_name, data, name=None):
    x, y = split_data(data)
    model = open_model(model_name)
    score = model.predict_proba(x)[:, 1]
    print("predicted")
    if y is not None:
        model_params = get_hyperparameters(model_name)
        model_params = calc_error(score, y, model_params, name)
        save_model(model_params, name=model_name)
    threshold = get_hyperparameters(model_name)["threshold"]
    pred = (score >= threshold).astype(int)
    result = pd.DataFrame({
        "Class": pred,
        "Score": score
    })
    return result