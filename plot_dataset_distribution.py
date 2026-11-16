import matplotlib.pyplot as plt
from pathlib import Path

print("=" * 60)
print("       AGRITRUST AI - DATASET VISUALIZATION")
print("=" * 60)

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(exist_ok=True)

extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

categories = []
image_counts = []

for folder in sorted(DATASET_DIR.iterdir()):

    if not folder.is_dir():
        continue

    if folder.name.lower() in ["plant_village", "paddy_doctor"]:
        continue

    count = sum(
        1 for file in folder.rglob("*")
        if file.is_file() and file.suffix.lower() in extensions
    )

    if count > 0:
        categories.append(folder.name)
        image_counts.append(count)

print("\nTotal Categories:", len(categories))
print("Total Images:", sum(image_counts))

# Create graph
plt.figure(figsize=(14, 16))

plt.barh(categories, image_counts)

plt.title(
    "AgriTrust AI - Disease Dataset Distribution",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Number of Images")
plt.ylabel("Disease Categories")

plt.grid(axis="x", linestyle="--", alpha=0.5)

plt.tight_layout()

# Save graph
output_path = RESULTS_DIR / "dataset_distribution.png"

plt.savefig(output_path, dpi=300, bbox_inches="tight")

plt.show()

print("\nGraph saved successfully!")
print("Location:", output_path)

print("\nDATASET VISUALIZATION COMPLETED!")
print("=" * 60)