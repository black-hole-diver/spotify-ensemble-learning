from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd

pred = pd.read_csv("results/q1_1_predictions.csv")

actual = pred["Actual"]
models = [col for col in pred.columns if col != "Actual"]

fig, axes = plt.subplots(2, 3, figsize=(14, 9))

for ax, name in zip(axes.flat, models):
    ConfusionMatrixDisplay.from_predictions(
        actual,
        pred[name],
        labels=sorted(actual.unique()),
        ax=ax,
        cmap="Blues",
        colorbar=False,
        xticks_rotation=45
    )
    ax.set_title(name)

plt.suptitle("Spotify Genre Classification Confusion Matrices")
plt.tight_layout()
plt.savefig("results/q1_1_confusion_matrices.png", dpi=300)
plt.show()
