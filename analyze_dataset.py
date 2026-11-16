import os

from pathlib import Path
from collections import defaultdict

print("=" * 60)
print("       AGRITRUST AI - DATASET ANALYSIS")
print("=" * 60)

# Dataset path
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"

# Supported image formats
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

if not DATASET_DIR.exists():
    print("ERROR: Dataset folder not found!")
    exit()

crop_data = defaultdict(lambda: defaultdict(int))

total_images = 0
total_categories = 0
empty_folders = []

print("\nDISEASE CATEGORY ANALYSIS")
print("-" * 60)

# Read category folders
for folder in sorted(DATASET_DIR.iterdir()):

    if not folder.is_dir():
        continue

    # Skip container folder
    if folder.name.lower() == "plant_village":
        continue

    image_count = sum(
        1 for file in folder.rglob("*")
        if file.is_file()
        and file.suffix.lower() in IMAGE_EXTENSIONS
    )

    if image_count == 0:
        empty_folders.append(folder.name)
        continue

    # Separate crop and disease names
    if "___" in folder.name:
        crop_name, disease_name = folder.name.split("___", 1)
    else:
        crop_name = "Other"
        disease_name = folder.name

    crop_data[crop_name][disease_name] += image_count

    total_images += image_count
    total_categories += 1

    print(f"\nCrop: {crop_name}")
    print(f"Disease: {disease_name}")
    print(f"Images: {image_count}")

print("\n" + "=" * 60)
print("FINAL DATASET SUMMARY")
print("=" * 60)

print("Total Crop Types:", len(crop_data))
print("Total Disease Categories:", total_categories)
print("Total Images:", total_images)

print("\nCROP-WISE IMAGE SUMMARY")
print("-" * 60)

for crop, diseases in crop_data.items():

    crop_total = sum(diseases.values())

    print(f"\n{crop}: {crop_total} images")

    for disease, count in diseases.items():
        print(f"   {disease}: {count}")

if empty_folders:
    print("\nEMPTY FOLDERS:")
    for folder in empty_folders:
        print(folder)

print("\n" + "=" * 60)
print("DATASET ANALYSIS COMPLETED!")
print("=" * 60)