from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data_loading import load_data

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
RANDOM_STATE = 42  # fixed seed so the results can be reproduced


def prepare_features(df: pd.DataFrame):
    """Split the table into input features (X) and the target label (y)."""
    X = df.drop(columns=["Class"])
    y = df["Class"]
    return X, y


def build_model():
    """Scale the features, then fit logistic regression.

    A pipeline keeps the scaling and the model together, so the scaler
    only learns from the training data and never sees the test data.
    """
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_STATE,
        ),
    )


if __name__ == "__main__":
    df = load_data()
    X, y = prepare_features(df)

    # stratify=y keeps the fraud rate the same in train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
    )
    print("Train size:", len(X_train), "| Test size:", len(X_test))
    print("Fraud cases in test set:", y_test.sum())

    model = build_model()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)                # 0 or 1 for each transaction
    y_score = model.predict_proba(X_test)[:, 1]   # probability of fraud

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=4))
    print("Average precision (PR-AUC):", round(average_precision_score(y_test, y_score), 4))

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Legitimate", "Fraud"])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, values_format="d", colorbar=False)
    ax.set_title("Baseline logistic regression: confusion matrix")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "baseline_confusion_matrix.png", dpi=150)
    print("\nSaved reports/figures/baseline_confusion_matrix.png")