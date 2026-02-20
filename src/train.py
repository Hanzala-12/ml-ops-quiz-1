from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


PROCESSED_DIR = Path("data/processed")
MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")
MODEL_PATH = MODEL_DIR / "logistic_regression.joblib"


def main() -> None:
    x_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    x_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")
    y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv").squeeze("columns")
    y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv").squeeze("columns")

    model = LogisticRegression(max_iter=2000, solver="liblinear", random_state=42)
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    with open(RESULTS_DIR / "train_accuracy.txt", "w", encoding="utf-8") as f:
        f.write(f"accuracy={accuracy:.4f}\n")

    print(f"Model saved to: {MODEL_PATH}")
    print(f"Training completed. Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
