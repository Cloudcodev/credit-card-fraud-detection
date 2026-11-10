from pathlib import Path

import pandas as pd

# find project root from this file's location so the script works from any folder
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    "Load the credit card transactions CSV into a Pandas DataFrame."
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. "
            "Download creditcard.csv from Kaggle and place it in data/raw/."
        )
    return pd.read_csv(path)


if __name__ == "__main__":
    df = load_data()

    print("Shape (rows, columns):", df.shape)
    print("\nColumns:", list(df.columns))
    print("\nClass counts (0 = legitimate, 1 = fraud):")
    print(df["Class"].value_counts())
    print(f"\nFraud rate: {df['Class'].mean():.4%}")