import pandas as pd

# Load fake news
fake = pd.read_csv("data/Fake.csv")
fake["label"] = 0   # Fake = 0

# Load real news
real = pd.read_csv("data/True.csv")
real["label"] = 1   # Real = 1

# Combine both
df = pd.concat([fake, real], ignore_index=True)

# Keep only required columns
df = df[["title", "text", "label"]]

# Save final dataset
df.to_csv("data/full_data.csv", index=False)

print("✅ full_data.csv created successfully")