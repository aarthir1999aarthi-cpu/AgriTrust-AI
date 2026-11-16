import pandas as pd
import os

# Project paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_PATH = os.path.join(
    BASE_DIR, "data", "raw", "agritrust_dataset.csv"
)

OUTPUT_DIR = os.path.join(BASE_DIR, "data", "processed")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load dataset
print("Loading Dataset...")

df = pd.read_csv(DATASET_PATH)

print("Dataset Loaded Successfully!")
print("Original Shape:", df.shape)

# Remove duplicate rows
df = df.drop_duplicates()

# Handle missing values
for column in df.columns:
    if df[column].dtype == "object":
        df[column] = df[column].fillna("Unknown")
    else:
        df[column] = df[column].fillna(df[column].median())

# Save processed dataset
output_path = os.path.join(
    OUTPUT_DIR, "processed_dataset.csv"
)

df.to_csv(output_path, index=False)

print("\nPreprocessing Completed Successfully!")
print("Final Shape:", df.shape)
print("Saved Location:", output_path)