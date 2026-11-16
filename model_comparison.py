import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import os

# ==========================================
# AGRITRUST AI - MODEL PERFORMANCE COMPARISON
# ==========================================

print("\nAGRITRUST AI - MODEL COMPARISON")
print("=" * 45)

# Step 1: Create results folder
os.makedirs("results", exist_ok=True)

# Step 2: Model names
models = ["CNN", "Decision Tree", "Random Forest"]

# Step 3: Actual evaluation results
accuracy = [89.11, 82.54, 92.30]
precision = [89.56, 82.74, 92.22]
recall = [89.11, 82.54, 92.30]
f1_score = [89.03, 82.60, 92.20]

print("\nModel results loaded successfully!")

# ==========================================
# GRAPH 1: ACCURACY COMPARISON
# ==========================================

print("\nGenerating accuracy comparison graph...")

plt.figure(figsize=(9, 5))

bars = plt.bar(models, accuracy)

plt.title("Accuracy Comparison of AgriTrust AI Models")
plt.xlabel("Machine Learning Models")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 100)
plt.grid(axis="y", linestyle="--", alpha=0.4)

for bar, value in zip(bars, accuracy):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 1,
        f"{value:.2f}%",
        ha="center"
    )

plt.tight_layout()
plt.savefig(
    "results/accuracy_comparison.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print("Accuracy graph saved successfully!")

# ==========================================
# GRAPH 2: OVERALL PERFORMANCE COMPARISON
# ==========================================

print("\nGenerating overall performance graph...")

x = np.arange(len(models))
width = 0.2

plt.figure(figsize=(11, 6))

plt.bar(x - 1.5 * width, accuracy, width, label="Accuracy")
plt.bar(x - 0.5 * width, precision, width, label="Precision")
plt.bar(x + 0.5 * width, recall, width, label="Recall")
plt.bar(x + 1.5 * width, f1_score, width, label="F1 Score")

plt.xticks(x, models)
plt.xlabel("Machine Learning Models")
plt.ylabel("Performance (%)")
plt.title("Overall Model Performance Comparison")
plt.ylim(0, 100)

plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig(
    "results/model_performance_comparison.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print("Performance comparison graph saved successfully!")

# ==========================================
# STEP 4: SAVE COMPARISON REPORT
# ==========================================

print("\nGenerating model comparison report...")

report_path = "results/model_comparison_report.txt"

with open(report_path, "w", encoding="utf-8") as file:

    file.write("AGRITRUST AI - MODEL PERFORMANCE COMPARISON\n")
    file.write("=" * 55 + "\n\n")

    for i, model in enumerate(models):

        file.write(f"MODEL: {model}\n")
        file.write("-" * 35 + "\n")

        file.write(f"Accuracy  : {accuracy[i]:.2f}%\n")
        file.write(f"Precision : {precision[i]:.2f}%\n")
        file.write(f"Recall    : {recall[i]:.2f}%\n")
        file.write(f"F1 Score  : {f1_score[i]:.2f}%\n\n")

    best_index = accuracy.index(max(accuracy))

    file.write("=" * 55 + "\n")
    file.write(f"BEST MODEL: {models[best_index]}\n")
    file.write(f"TEST ACCURACY: {accuracy[best_index]:.2f}%\n")
    file.write("=" * 55 + "\n")

print("Comparison report saved successfully!")

# ==========================================
# STEP 5: FINAL STATUS
# ==========================================

print("\n" + "=" * 50)
print("MODEL COMPARISON COMPLETED SUCCESSFULLY!")
print("=" * 50)

print("\nGenerated files:")

print("1. results/accuracy_comparison.png")
print("2. results/model_performance_comparison.png")
print("3. results/model_comparison_report.txt")

print("\nBest Model: Random Forest")
print("Test Accuracy: 92.30%")

print("\nAgriTrust AI comparison stage completed!")