from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


RAW_DATA_PATH = Path("data/raw/breast_cancer.csv")
PROCESSED_DIR = Path("data/processed")
TARGET_COLUMN = "target"
TEST_SIZE = 0.2
RANDOM_STATE = 42


def main() -> None:
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {RAW_DATA_PATH}")

    df = pd.read_csv(RAW_DATA_PATH)
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Expected target column '{TARGET_COLUMN}' in dataset")

    features = df.drop(columns=[TARGET_COLUMN])
    target = df[TARGET_COLUMN]

    # Fill numeric columns with median to handle missing values.
    features = features.fillna(features.median(numeric_only=True))

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    x_train.to_csv(PROCESSED_DIR / "X_train.csv", index=False)
    x_test.to_csv(PROCESSED_DIR / "X_test.csv", index=False)
    y_train.to_csv(PROCESSED_DIR / "y_train.csv", index=False)
    y_test.to_csv(PROCESSED_DIR / "y_test.csv", index=False)

    print(f"Loaded dataset shape: {df.shape}")
    print(f"Missing values before fill: {int(df.isna().sum().sum())}")
    print(f"Training samples: {len(x_train)}")
    print(f"Testing samples: {len(x_test)}")
    print(f"Processed files saved to: {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
