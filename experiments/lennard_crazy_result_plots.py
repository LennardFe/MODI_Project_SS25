import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix

# Real-life values
ground_truth = [
    "5C19", "DC0F", "96BB", "5C19", "DC0F",
    "96BB", "DC0F", "5C19", "DC0F", "5C19",
    "96BB", "5C19", "5C19", "DC0F", "96BB",
    "96BB", "DC0F", "96BB", "DC0F", "5C19"
]

# Base folder
folder = "results"

# Experiment ID to analyze
experiment_id = "43"  # e.g., "1", "2", "3", "41", etc.

# Containers
accuracies = {
    "selection": [],
    "distance_change": [],
    "bearing": []
}
all_preds = {
    "selection": [],
    "distance_change": [],
    "bearing": []
}
all_true = []

# Filter and read files
for file in os.listdir(folder):
    if not file.endswith(".csv"):
        continue
    if not file.startswith(f"{experiment_id}_"):
        continue  # Skip if not matching experiment

    df = pd.read_csv(os.path.join(folder, file))

    # Extend ground truth once per file
    all_true.extend(ground_truth)

    for col in ["selection", "distance_change", "bearing"]:
        all_preds[col].extend(df[col].tolist())
        acc = accuracy_score(ground_truth, df[col])
        accuracies[col].append(acc)

# Output average accuracy
print(f"\nResults for Experiment {experiment_id}")
print("Average Accuracy per Algorithm:")
for k, v in accuracies.items():
    print(f"  {k:16}: {sum(v)/len(v):.2%} (based on {len(v)} files)")

# Accuracy Barplot
plt.figure(figsize=(8, 5))
avg_acc = {k: sum(v)/len(v) for k, v in accuracies.items()}
sns.barplot(x=list(avg_acc.keys()), y=list(avg_acc.values()))
plt.ylim(0, 1)
plt.ylabel("Accuracy")
plt.title(f"Average Accuracy - Experiment {experiment_id}")
plt.show()

# Per-step accuracy heatmap
per_step_correct = {
    alg: [1 if pred == gt else 0 for pred, gt in zip(all_preds[alg], all_true)]
    for alg in ["selection", "distance_change", "bearing"]
}
df_per_step = pd.DataFrame(per_step_correct)
df_per_step["step"] = df_per_step.index % 20
heatmap_data = df_per_step.groupby("step").mean()

plt.figure(figsize=(10, 4))
sns.heatmap(heatmap_data.T, annot=True, cmap="YlGnBu", cbar=True)
plt.title(f"Per-Step Accuracy Heatmap - Experiment {experiment_id}")
plt.xlabel("Step (0–19)")
plt.ylabel("Algorithm")
plt.show()

# Confusion Matrices
for alg in ["selection", "distance_change", "bearing"]:
    cm = confusion_matrix(all_true, all_preds[alg], labels=["5C19", "DC0F", "96BB"])
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=["5C19", "DC0F", "96BB"],
                yticklabels=["5C19", "DC0F", "96BB"], cmap="Blues")
    plt.title(f"Confusion Matrix - {alg} - Experiment {experiment_id}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
