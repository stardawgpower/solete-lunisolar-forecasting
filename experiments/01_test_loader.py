from pathlib import Path
from src.data_loader import load_solete_60min

dataset_path = Path("solete_dataset/SOLETE_Pombo_60min.h5")

df = load_solete_60min(dataset_path)

print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nDate range:")
print(df.index.min(), "to", df.index.max())

print("\nMissing values:")
print(df.isna().sum())

print("\nDuplicate timestamps:")
print(df.index.duplicated().sum())

print("\nTime differences:")
print(df.index.to_series().diff().value_counts().head())

print("\nBasic statistics:")
print(df.describe().T)