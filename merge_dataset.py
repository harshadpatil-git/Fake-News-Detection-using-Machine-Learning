import pandas as pd

# Load datasets
fake = pd.read_csv("dataset/Fake.csv")
true = pd.read_csv("dataset/True.csv")

# Add labels
fake["label"] = "FAKE"
true["label"] = "REAL"

# Combine datasets
data = pd.concat([fake, true], axis=0)

# Keep only required columns
data = data[["text", "label"]]

# Shuffle dataset
data = data.sample(frac=1)

# Save merged dataset
data.to_csv("dataset/news.csv", index=False)

print("Datasets merged successfully!")