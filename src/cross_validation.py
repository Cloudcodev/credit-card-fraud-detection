from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

from baseline_model import PROJECT_ROOT, RANDOM_STATE, build_model, prepare_features
from boosted_model import build_boosted_model
from data_loading import load_data

RESULTS_PATH = PROJECT_ROOT / "reports" / "cv_results.csv"


if __name__ == "__main__":
    df = load_data()
    X, y = prepare_features(df)

    # 5 folds, shuffled with a fixed seed, keeping the fraud rate the same in each fold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    rows = []
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        # iloc selects rows by position, which is what the fold indices contain
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        baseline = build_model().fit(X_train, y_train)
        baseline_score = average_precision_score(y_test, baseline.predict_proba(X_test)[:, 1])

        boosted = build_boosted_model().fit(X_train, y_train)
        boosted_score = average_precision_score(y_test, boosted.predict_proba(X_test)[:, 1])

        rows.append({
            "fold": fold,
            "test_fraud_cases": int(y_test.sum()),
            "baseline_pr_auc": round(baseline_score, 4),
            "boosted_pr_auc": round(boosted_score, 4),
            "difference": round(boosted_score - baseline_score, 4),
        })
        print(rows[-1])

    results = pd.DataFrame(rows)
    results.to_csv(RESULTS_PATH, index=False)

    print("\nSummary across folds (mean and standard deviation):")
    print(results[["baseline_pr_auc", "boosted_pr_auc", "difference"]].agg(["mean", "std"]).round(4))
    print("\nSaved reports/cv_results.csv")