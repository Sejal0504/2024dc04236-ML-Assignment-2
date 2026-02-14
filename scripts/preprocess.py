import pandas as pd
from sklearn.model_selection import train_test_split

# File paths
RAW = r"/home/cloud/ML/data/raw/adult.csv"
TRAIN = r"/home/cloud/ML/data/processed/train.csv"
TEST = r"/home/cloud/ML/data/processed/test.csv"

print("Loading dataset...")
df = pd.read_csv(RAW)

# CLEANING

print("Cleaning data...")

# Replace ? with missing and drop
df = df.replace("?", pd.NA)
df = df.dropna()

# Remove extra spaces
for col in df.select_dtypes(include="object"):
    df[col] = df[col].str.strip()

# SPLIT

print("Splitting data...")

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["income"]
)

# Save files
train_df.to_csv(TRAIN, index=False)
test_df.to_csv(TEST, index=False)

print("Data cleaned & saved!")
print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)


