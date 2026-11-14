import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/q1_1_results.csv")

print("Available columns: ", df.columns.tolist())
print(df.head())

metrics = ["Accuracy", "F1 Score", "Macro F1 Score"]
x=np.arange(len(df))
width = 0.2

fig, ax = plt.subplots(figsize=(10, 6))
for i, metric in enumerate(metrics):
    ax.bar(x + (i-1) * width, df[metric]*100, width, label=metric)

ax.set_xticks(x)
ax.set_xticklabels(df["Classifier"], rotation=20, ha="right")
ax.set_ylabel("Percentage (%)")
ax.set_ylim(0, 100)
ax.set_title("Model Performance Comparison")
ax.legend()
ax.grid(axis="y", alpha=0.7)
plt.tight_layout()
plt.savefig("results/q1_1_comparison.png", dpi=300, bbox_inches="tight")
plt.show()