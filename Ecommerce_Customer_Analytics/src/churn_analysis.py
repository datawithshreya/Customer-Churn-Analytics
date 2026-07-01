"""
churn_analysis.py
-----------------
End-to-end churn prediction pipeline:

  1. Feature preparation  – encode categoricals, select predictors
  2. Train / test split   – stratified 80/20
  3. Random Forest model  – with class_weight='balanced' for imbalance
  4. Evaluation           – accuracy, precision, recall, F1, AUC
  5. Confusion matrix     – printed as ASCII grid
  6. Feature importance   – top 10 drivers
  7. Correlation analysis – Pearson on numeric features
  8. Hypothesis test      – t-test: churned vs retained monetary spend
"""

import pandas as pd
import numpy as np
from scipy import stats

from sklearn.model_selection  import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble         import RandomForestClassifier
from sklearn.preprocessing    import LabelEncoder
from sklearn.metrics          import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report,
)

RANDOM_STATE = 42

# ─────────────────────────────────────────────────────────────
# FEATURE PREP
# ─────────────────────────────────────────────────────────────

NUMERIC_FEATURES = [
    "age", "recency_days", "frequency", "monetary",
    "avg_order_value", "purchase_frequency", "customer_lifetime",
    "clv", "loyalty_points", "avg_review_rating", "total_reviews",
]
CATEGORICAL_FEATURES = [
    "gender", "customer_segment", "favourite_category", "preferred_channel",
]
TARGET = "churned"


def prepare_features(feat: pd.DataFrame):
    """
    Encode categoricals, impute, return (X, y, feature_names).
    """
    df = feat.copy()

    # Label-encode each categorical
    le = LabelEncoder()
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str).fillna("Unknown"))

    # Fill any residual NaN in numeric cols
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    all_features = [c for c in NUMERIC_FEATURES + CATEGORICAL_FEATURES if c in df.columns]
    X = df[all_features].copy()
    y = df[TARGET].astype(int)

    return X, y, all_features


# ─────────────────────────────────────────────────────────────
# TRAIN MODEL
# ─────────────────────────────────────────────────────────────

def train_churn_model(feat: pd.DataFrame):
    """
    Returns (model, X_test, y_test, y_pred, y_prob, feature_names)
    """
    X, y, feature_names = prepare_features(feat)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators      = 200,
        max_depth         = 8,
        min_samples_split = 10,
        class_weight      = "balanced",
        random_state      = RANDOM_STATE,
        n_jobs            = -1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return model, X_train, X_test, y_train, y_test, y_pred, y_prob, feature_names


# ─────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────

def evaluate_model(y_test, y_pred, y_prob) -> dict:
    metrics = {
        "accuracy":  round(accuracy_score(y_test, y_pred),  4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred,    zero_division=0), 4),
        "f1":        round(f1_score(y_test, y_pred,        zero_division=0), 4),
        "roc_auc":   round(roc_auc_score(y_test, y_prob),  4),
    }
    cm = confusion_matrix(y_test, y_pred)
    return metrics, cm


def _ascii_confusion_matrix(cm) -> str:
    tn, fp, fn, tp = cm.ravel()
    lines = [
        "                  Predicted",
        "                  No-Churn  Churned",
        f"  Actual No-Churn  {tn:>6}   {fp:>6}",
        f"  Actual Churned   {fn:>6}   {tp:>6}",
    ]
    return "\n".join(lines)


def get_feature_importance(model, feature_names: list) -> pd.DataFrame:
    imp = pd.DataFrame({
        "feature":   feature_names,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    return imp


# ─────────────────────────────────────────────────────────────
# STATISTICAL TESTS
# ─────────────────────────────────────────────────────────────

def correlation_analysis(feat: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in NUMERIC_FEATURES if c in feat.columns] + ["churned"]
    corr = feat[cols].corr()[["churned"]].drop("churned").sort_values("churned")
    corr.columns = ["correlation_with_churn"]
    corr["correlation_with_churn"] = corr["correlation_with_churn"].round(4)
    return corr


def hypothesis_test(feat: pd.DataFrame) -> dict:
    """
    H0: mean monetary spend is equal for churned vs retained customers.
    H1: mean monetary spend differs.
    """
    churned   = feat[feat["churned"] == 1]["monetary"].dropna()
    retained  = feat[feat["churned"] == 0]["monetary"].dropna()
    t_stat, p_val = stats.ttest_ind(churned, retained, equal_var=False)
    return {
        "test":       "Welch's t-test (monetary spend: churned vs retained)",
        "t_statistic": round(t_stat, 4),
        "p_value":     round(p_val,  6),
        "significant": p_val < 0.05,
        "churned_mean":    round(churned.mean(), 2),
        "retained_mean":   round(retained.mean(), 2),
    }


# ─────────────────────────────────────────────────────────────
# PRINT RESULTS
# ─────────────────────────────────────────────────────────────

def print_churn_results(metrics, cm, imp, corr, hyp) -> None:
    print("\n--- Churn Prediction Model (Random Forest) ----------------")
    print(f"\n  Accuracy   : {metrics['accuracy']*100:.2f}%")
    print(f"  Precision  : {metrics['precision']*100:.2f}%")
    print(f"  Recall     : {metrics['recall']*100:.2f}%")
    print(f"  F1 Score   : {metrics['f1']*100:.2f}%")
    print(f"  ROC-AUC    : {metrics['roc_auc']:.4f}")

    print("\n  Confusion Matrix:")
    print(_ascii_confusion_matrix(cm))

    print("\n  Top 10 Feature Importances:")
    for _, row in imp.head(10).iterrows():
        bar = "▓" * int(row["importance"] * 100)
        bar = "#" * int(row['importance'] * 50)
    print("\n--- Correlation Analysis (vs Churn) ----------------")
    for feat_name, row in corr.iterrows():
        direction = "↑" if row["correlation_with_churn"] > 0 else "-"
        direction = "+" if row["correlation_with_churn"] > 0 else "-"
    print("\n--- Hypothesis Test ----------------")
    print(f"  {hyp['test']}")
    print(f"  Churned mean spend   : ${hyp['churned_mean']:,.2f}")
    print(f"  Retained mean spend  : ${hyp['retained_mean']:,.2f}")
    print(f"  t-statistic          : {hyp['t_statistic']}")
    print(f"  p-value              : {hyp['p_value']}")
    sig = " Significant (reject H0)" if hyp["significant"] else " Not significant (fail to reject H0)"
    print(f"  Result               : {sig}")
    print()
