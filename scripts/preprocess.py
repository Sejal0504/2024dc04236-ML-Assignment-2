import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

# Path
BASE_DIR = Path(__file__).resolve().parent.parent
RAW = BASE_DIR / "data" / "raw" / "adult.csv"
TRAIN = BASE_DIR / "data" / "processed" / "train.csv"
TEST = BASE_DIR / "data" / "processed" / "test.csv"

# Create processed folder if not exists
TRAIN.parent.mkdir(parents=True, exist_ok=True)

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

print("\nSaved files:")
print("Train:", TRAIN)
print("Test:", TEST)

