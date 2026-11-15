from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

from baseline_model import FIGURES_DIR, RANDOM_STATE, build_model, prepare_features
from data_loading import load_data


def build_boosted_model():
    """Gradient boosting with balanced class weights, matching the baseline's treatment of fraud."""
    return HistGradientBoostingClassifier(
        class_weight="balanced",
        max_iter=200,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
    )


if __name__ == "__main__":
    df = load_data()
    X, y = prepare_features(df)

    # same split as the baseline so the two scores are directly comparable
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
    )

    # baseline score on this split, recomputed here for a fair side-by-side
    baseline = build_model().fit(X_train, y_train)
    baseline_pr_auc = average_precision_score(y_test, baseline.predict_proba(X_test)[:, 1])

    boosted = build_boosted_model().fit(X_train, y_train)
    y_pred = boosted.predict(X_test)
    y_score = boosted.predict_proba(X_test)[:, 1]
    boosted_pr_auc = average_precision_score(y_test, y_score)

    print("Classification report (gradient boosting):")
    print(classification_report(y_test, y_pred, digits=4))
    print("Baseline PR-AUC (logistic regression):", round(baseline_pr_auc, 4))
    print("Gradient boosting PR-AUC:", round(boosted_pr_auc, 4))

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Legitimate", "Fraud"])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, values_format="d", colorbar=False)
    ax.set_title("Gradient boosting: confusion matrix")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "gradient_boosting_confusion_matrix.png", dpi=150)
    print("\nSaved reports/figures/gradient_boosting_confusion_matrix.png")