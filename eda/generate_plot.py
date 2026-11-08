import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.metrics import (precision_recall_curve, average_precision_score,
                             confusion_matrix, ConfusionMatrixDisplay, f1_score)

y = pd.read_csv("eda/true.csv")["Class"].to_numpy()
p = pd.read_csv("eda/pred.csv")
score, label = p["Score"].to_numpy(), p["Class"].to_numpy()
thr = score[label == 1].min()

prec, rec, ths = precision_recall_curve(y, score)
f1 = 2 * prec * rec / (prec + rec + 1e-10)
ap = average_precision_score(y, score)

fig, ax = plt.subplots(2, 2, figsize=(12, 9), dpi=100)

ax[0, 0].plot(rec, prec, color="#1f77b4", label=f"AP = {ap:.3f}")
ax[0, 0].axhline(y.mean(), color="red", ls="--", label=f"Random = {y.mean():.4f}")
ax[0, 0].set(title="Precision-Recall Curve", xlabel="Recall", ylabel="Precision")
ax[0, 0].legend()

ConfusionMatrixDisplay(confusion_matrix(y, label), display_labels=["Normal", "Fraud"]).plot(
    ax=ax[0, 1], colorbar=False, cmap="Blues")
ax[0, 1].set_title(f"Confusion Matrix (F1 = {f1_score(y, label):.3f})")

ax[1, 0].hist(score[y == 0], bins=50, alpha=0.6, label="Normal", color="#1f77b4")
ax[1, 0].hist(score[y == 1], bins=50, alpha=0.6, label="Fraud", color="#d62728")
ax[1, 0].axvline(thr, color="k", ls="--", label=f"Threshold = {thr:.2f}")
ax[1, 0].set(yscale="log", title="Score Distribution by Class", xlabel="Predicted probability", ylabel="quantity")
ax[1, 0].legend()

ax[1, 1].plot(ths, prec[:-1], label="Precision")
ax[1, 1].plot(ths, rec[:-1], label="Recall")
ax[1, 1].plot(ths, f1[:-1], label="F1", lw=2)
ax[1, 1].axvline(thr, color="k", ls="--")
ax[1, 1].set(title="Metrics vs Threshold", xlabel="Threshold")
ax[1, 1].legend()

for a in ax.ravel(): a.grid(ls=":", alpha=0.5)
plt.tight_layout()
plt.savefig("eda/evaluation_plot.png", dpi=200)
plt.show()