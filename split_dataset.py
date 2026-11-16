import random
import shutil
from pathlib import Path

SOURCE_DIR = Path("dataset")
OUTPUT_DIR = Path("dataset_split_v2")

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

random.seed(42)


def split_dataset():

    if not SOURCE_DIR.exists():
        print("Dataset folder not found!")
        return

    if OUTPUT_DIR.exists():
        print("dataset_split_v2 already exists.")
        print("Rename or remove it before running again.")
        return

    total_images = 0
    total_classes = 0

    for class_folder in sorted(SOURCE_DIR.iterdir()):

        if not class_folder.is_dir():
            continue

        # Search images inside all nested folders
        images = [
            file for file in class_folder.rglob("*")
            if file.is_file()
            and file.suffix.lower() in IMAGE_EXTENSIONS
        ]

        if not images:
            print(f"Skipping empty folder: {class_folder.name}")
            continue

        random.shuffle(images)

        total = len(images)

        train_count = int(total * TRAIN_RATIO)
        val_count = int(total * VAL_RATIO)

        train_images = images[:train_count]
        val_images = images[train_count:train_count + val_count]
        test_images = images[train_count + val_count:]

        splits = {
            "train": train_images,
            "validation": val_images,
            "test": test_images
        }

        for split_name, split_images in splits.items():

            destination = OUTPUT_DIR / split_name / class_folder.name
            destination.mkdir(parents=True, exist_ok=True)

            for index, image in enumerate(split_images):

                # Preserve unique filenames
                new_name = f"{index}_{image.name}"

                shutil.copy2(
                    image,
                    destination / new_name
                )

        total_images += total
        total_classes += 1

        print(
            f"{class_folder.name}: "
            f"Total={total}, "
            f"Train={len(train_images)}, "
            f"Validation={len(val_images)}, "
            f"Test={len(test_images)}"
        )

    print("\n================================")
    print("CORRECTED DATASET SPLITTING DONE")
    print("================================")
    print("Total Categories:", total_classes)
    print("Total Images:", total_images)
    print("Output Folder:", OUTPUT_DIR)


if __name__ == "__main__":
    split_dataset()