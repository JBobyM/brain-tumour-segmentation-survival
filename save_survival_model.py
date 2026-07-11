"""Fit the survival RandomForest on all cases and persist it for the demo.

Uses the end-to-end feature set (clinical + Phase-2 predicted-mask features),
so the saved model consumes exactly what the app can compute at inference.
"""

import joblib
import pandas as pd

from train_survival import CLINICAL, make_model, CLASSES

OUT = "survival_model.joblib"


def main():
    df = pd.read_csv("survival_features.csv")
    df = df[df["survival_class"].notna()].reset_index(drop=True)
    pred_cols = [c for c in df.columns if c.startswith("pred_")]
    feat_cols = CLINICAL + pred_cols

    X = df[feat_cols].fillna(0.0).values
    y = df["survival_class"].astype(int).values
    model = make_model().fit(X, y)

    joblib.dump({"model": model, "feature_cols": feat_cols, "classes": CLASSES}, OUT)
    print(f"[save] fitted RF on {len(df)} cases, {len(feat_cols)} features -> {OUT}")
    print(f"[save] features: {feat_cols}")


if __name__ == "__main__":
    main()
