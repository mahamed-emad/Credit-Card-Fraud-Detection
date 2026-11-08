from time import time
from src.train import *
from src.predict import *

model_params = {
    "models": "Binary_Classification",
    "max_iter": int(1e4),
    "alpha": 0.01,
    "hidden_layer_sizes": (64, 32),
    "activation": 'relu',
    "solver":'adam',
    "batch_size": 32,
    "techniques": [[0.08]],
    "tol": 5e-4,
    "random_state": 42,
    "early_stopping": True,
    "validation_fraction": 0.05,
    "learning_rate_init": 0.001
}

start = time()
model_name = "model0"
model_name = train(model_params, param_grid={"model__alpha": model_params["techniques"][0]})
calc_threshold(model_name, read("train"))

test(model_name, read("train"), "train")
test(model_name, read("val"), "val")

pred = test(model_name, read("test"), "test")
save(pred, "eda/pred", change=True)
save(read("test")[["Class"]], "eda/true", change=True)

print(f"TIME-TAKEN: {(time() - start) / 60:.1f} mins.")