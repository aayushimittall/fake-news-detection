import pandas as pd
from sklearn.model_selection import train_test_split
import os

print("✅ split_data.py started")

# Check full_data.csv existence
if not os.path.exists("data/full_data.csv"):
    raise FileNotFoundError("❌ data/full_data.csv not found")

print("✅ full_data.csv found")

# Load data
df = pd.read_csv("data/full_data.csv")
print("✅ full_data.csv loaded")
print("✅ Shape:", df.shape)
print("✅ Columns:", df.columns.tolist())

# Validate columns
required_cols = {"title", "text", "label"}
if not required_cols.issubset(df.columns):
    raise ValueError("❌ full_data.csv missing required columns")

# Split
train_df, temp_df = train_test_split(
    df, test_size=0.2, stratify=df["label"], random_state=42
)

valid_df, test_df = train_test_split(
    temp_df, test_size=0.5, stratify=temp_df["label"], random_state=42
)

# Save
train_df.to_csv("data/train.csv", index=False)
valid_df.to_csv("data/valid.csv", index=False)
test_df.to_csv("data/test.csv", index=False)

print("🎉 DONE")
print("train.csv rows:", len(train_df))
print("valid.csv rows:", len(valid_df))
print("test.csv rows:", len(test_df))