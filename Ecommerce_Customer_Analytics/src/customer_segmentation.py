"""
customer_segmentation.py
------------------------
Two complementary segmentation approaches:

  1. RFM Scoring  – rule-based quintile scoring → named segments
  2. K-Means      – unsupervised clustering on scaled RFM features

Segment labels (both approaches converge on similar groups):
  Champions       – high R, F, M
  Loyal Customers – high F & M, moderate R
  Potential       – moderate overall
  At Risk         – previously active, now declining
  Lost            – low R, F, M
  New Customers   – low F but recent
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

N_CLUSTERS = 5   # tunable

# ─────────────────────────────────────────────────────────────
# 1. RFM SCORE-BASED SEGMENTATION
# ─────────────────────────────────────────────────────────────

def _rfm_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign 1-5 scores for R, F, M using quintile binning.
    Recency is inverted (lower recency = better = score 5).
    """
    df = df.copy()

    # Protect against all-zero edge cases
    def safe_qcut(series, q, labels, **kwargs):
        try:
            return pd.qcut(series, q=q, labels=labels, duplicates="drop", **kwargs)
        except ValueError:
            return pd.cut(series, bins=q, labels=labels[:q-1], duplicates="drop")

    # R score: 5 = most recent (lowest recency_days)
    df["R_score"] = safe_qcut(
        df["recency_days"], 5, labels=[5, 4, 3, 2, 1]
    ).astype(int)

    # F score
    df["F_score"] = safe_qcut(
        df["frequency"].clip(lower=0), 5, labels=[1, 2, 3, 4, 5]
    ).astype(int)

    # M score
    df["M_score"] = safe_qcut(
        df["monetary"].clip(lower=0), 5, labels=[1, 2, 3, 4, 5]
    ).astype(int)

    df["RFM_score"] = df["R_score"] + df["F_score"] + df["M_score"]
    return df


def _assign_rfm_segment(row) -> str:
    r, f, m = row["R_score"], row["F_score"], row["M_score"]
    score   = row["RFM_score"]

    if r >= 4 and f >= 4 and m >= 4:
        return "Champions"
    elif f >= 4 and m >= 4:
        return "Loyal Customers"
    elif r >= 4 and f <= 2:
        return "New Customers"
    elif r <= 2 and f >= 3:
        return "At Risk"
    elif score <= 5:
        return "Lost"
    else:
        return "Potential Loyalists"


def rfm_segmentation(feat: pd.DataFrame) -> pd.DataFrame:
    feat = _rfm_score(feat)
    feat["rfm_segment"] = feat.apply(_assign_rfm_segment, axis=1)
    return feat


# ─────────────────────────────────────────────────────────────
# 2. K-MEANS CLUSTERING
# ─────────────────────────────────────────────────────────────

# Human-readable labels we assign after inspecting cluster centroids
_CLUSTER_LABEL_MAP = {
    0: "Potential Loyalists",
    1: "At Risk",
    2: "Champions",
    3: "Lost",
    4: "New Customers",
}


def kmeans_segmentation(feat: pd.DataFrame, n_clusters: int = N_CLUSTERS) -> pd.DataFrame:
    feat = feat.copy()

    rfm_cols = ["recency_days", "frequency", "monetary"]
    X        = feat[rfm_cols].fillna(0).copy()

    # Log-transform monetary & frequency to reduce skew
    X["frequency"] = np.log1p(X["frequency"])
    X["monetary"]  = np.log1p(X["monetary"])
    # Invert recency so that higher = more recent
    X["recency_days"] = np.log1p(X["recency_days"].max() - X["recency_days"])

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    feat["kmeans_cluster"] = km.fit_predict(X_scaled)

    # Map cluster numbers to interpretable labels
    # Use centroid R/F/M medians to auto-assign labels
    centroids = (
        feat.groupby("kmeans_cluster")[["recency_days", "frequency", "monetary"]]
        .median()
        .sort_values("monetary", ascending=False)
        .reset_index()
    )

    label_names = ["Champions", "Loyal Customers", "Potential Loyalists", "At Risk", "Lost"]
    cluster_label_map = dict(zip(centroids["kmeans_cluster"], label_names[:n_clusters]))
    feat["kmeans_segment"] = feat["kmeans_cluster"].map(cluster_label_map)

    return feat, km, scaler


# ─────────────────────────────────────────────────────────────
# MASTER
# ─────────────────────────────────────────────────────────────

def segment_customers(feat: pd.DataFrame):
    """
    Runs both RFM and K-Means segmentation.
    Returns (enriched_feat, kmeans_model, scaler).
    """
    feat        = rfm_segmentation(feat)
    feat, km, scaler = kmeans_segmentation(feat)
    return feat, km, scaler


def print_segmentation_results(feat: pd.DataFrame) -> None:
    print("\n--- Customer Segmentation Results ----------------")

    # RFM segments
    rfm_counts = feat["rfm_segment"].value_counts()
    print("\n  RFM Segments:")
    for seg, cnt in rfm_counts.items():
        pct = cnt / len(feat) * 100
        bar = "█" * int(pct / 2)
        bar = "#" * int(pct / 5)
    # K-Means segments
    km_counts = feat["kmeans_segment"].value_counts()
    print("\n  K-Means Segments:")
    for seg, cnt in km_counts.items():
        pct = cnt / len(feat) * 100
        bar = "█" * int(pct / 2)
        bar = "#" * int(pct / 5)
    # CLV by segment
    clv_by_seg = (
        feat.groupby("rfm_segment")["clv"]
        .agg(["mean","sum"])
        .sort_values("mean", ascending=False)
    )
    print("\n  Avg CLV by RFM Segment:")
    for seg, row in clv_by_seg.iterrows():
        print(f"    {seg:<22}  avg=${row['mean']:>8,.2f}   total=${row['sum']:>12,.2f}")
    print()
