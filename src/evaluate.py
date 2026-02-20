from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report


PROCESSED_DIR = Path("data/processed")
MODEL_PATH = Path("models/logistic_regression.joblib")
RESULTS_PATH = Path("results/evaluation.txt")


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model file missing. Run training first: make train"
        )

    model = joblib.load(MODEL_PATH)
    x_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")
    y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv").squeeze("columns")

    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {accuracy:.4f}\n\n")
        f.write(report)

    print(f"Evaluation completed. Accuracy: {accuracy:.4f}")
    print(f"Detailed report saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
