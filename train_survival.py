"""Phase 3, step 2 — 3-class survival prediction from tumour features.

Compares three feature sets with stratified 5-fold cross-validation:
    clinical_only        -> age + resection            (baseline)
    predicted (e2e)      -> clinical + Phase-2 predicted-mask features
    groundtruth (bound)  -> clinical + expert-mask features

Reports accuracy + macro one-vs-rest ROC-AUC (per the proposal), for both the
full cohort and the GTR-only subset (BraTS challenge protocol). Saves confusion
matrix, ROC curves, and feature-importance plots.
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

CSV = "survival_features.csv"
CLASSES = ["short (<10mo)", "mid (10-15mo)", "long (>15mo)"]
CLINICAL = ["age", "resection_gtr", "resection_known"]
SEED = 42


def make_model():
    return RandomForestClassifier(
        n_estimators=400, max_depth=5, min_samples_leaf=3,
        class_weight="balanced", random_state=SEED, n_jobs=-1,
    )


def evaluate(X, y, label):
    """Stratified 5-fold out-of-fold accuracy + macro OVR AUC."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    proba = cross_val_predict(make_model(), X, y, cv=skf, method="predict_proba", n_jobs=-1)
    preds = proba.argmax(1)
    acc = accuracy_score(y, preds)
    # macro one-vs-rest AUC (guard against a class missing in y)
    present = np.unique(y)
    try:
        auc = roc_auc_score(y, proba, multi_class="ovr", average="macro", labels=present)
    except Exception:
        auc = float("nan")
    return {"label": label, "n": len(y), "acc": acc, "auc": auc, "proba": proba, "preds": preds, "y": y}


def feature_sets(df):
    pred = [c for c in df.columns if c.startswith("pred_")]
    gt = [c for c in df.columns if c.startswith("gt_")]
    return {
        "clinical_only": CLINICAL,
        "predicted (end-to-end)": CLINICAL + pred,
        "groundtruth (upper bound)": CLINICAL + gt,
    }


def run_cohort(df, sets, cohort_name):
    y = df["survival_class"].astype(int).values
    print(f"\n===== {cohort_name}  (n={len(df)}) =====")
    print("class counts:", {CLASSES[k]: int((y == k).sum()) for k in range(3)})
    results = []
    for name, cols in sets.items():
        X = df[cols].fillna(0.0).values
        res = evaluate(X, y, name)
        results.append(res)
        print(f"  {name:28s}  acc={res['acc']:.3f}  macroAUC={res['auc']:.3f}  ({len(cols)} feats)")
    return results


def plot_confusion(res, path):
    cm = confusion_matrix(res["y"], res["preds"])
    fig, ax = plt.subplots(figsize=(5, 4.2))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(["short", "mid", "long"]); ax.set_yticklabels(["short", "mid", "long"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_title(f"Confusion — {res['label']}\nacc={res['acc']:.3f}  macroAUC={res['auc']:.3f}")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def plot_roc(res, path):
    y, proba = res["y"], res["proba"]
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    for k in range(3):
        fpr, tpr, _ = roc_curve((y == k).astype(int), proba[:, k])
        a = roc_auc_score((y == k).astype(int), proba[:, k])
        ax.plot(fpr, tpr, label=f"{CLASSES[k]} (AUC={a:.2f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(f"One-vs-Rest ROC — {res['label']}"); ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def plot_importance(df, cols, path, title):
    y = df["survival_class"].astype(int).values
    X = df[cols].fillna(0.0).values
    m = make_model().fit(X, y)
    imp = pd.Series(m.feature_importances_, index=cols).sort_values(ascending=True).tail(15)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(imp.index, imp.values, color="#4C78A8")
    ax.set_title(title); ax.set_xlabel("Random-forest importance")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wandb", action="store_true", help="log to Weights & Biases")
    args = ap.parse_args()

    df = pd.read_csv(CSV)
    df = df[df["survival_class"].notna()].reset_index(drop=True)
    sets = feature_sets(df)

    all_res = run_cohort(df, sets, "FULL COHORT")
    gtr = df[df["resection_gtr"] == 1].reset_index(drop=True)
    gtr_res = run_cohort(gtr, sets, "GTR-ONLY (challenge protocol)")

    # plots for the end-to-end model on the full cohort
    e2e = next(r for r in all_res if r["label"].startswith("predicted"))
    plot_confusion(e2e, "survival_confusion_matrix.png")
    plot_roc(e2e, "survival_roc_curves.png")
    plot_importance(df, sets["predicted (end-to-end)"], "survival_feature_importance.png",
                    "Survival — feature importance (end-to-end)")
    print("\n[survival] saved: survival_confusion_matrix.png, survival_roc_curves.png, survival_feature_importance.png")

    if args.wandb:
        import wandb
        wandb.init(project="brain-tumour-survival", name="rf-3class",
                   config={"model": "RandomForest", "cv": "stratified-5fold", "seed": SEED})
        for res in all_res:
            wandb.summary[f"full/{res['label']}/acc"] = res["acc"]
            wandb.summary[f"full/{res['label']}/auc"] = res["auc"]
        for res in gtr_res:
            wandb.summary[f"gtr/{res['label']}/acc"] = res["acc"]
            wandb.summary[f"gtr/{res['label']}/auc"] = res["auc"]
        wandb.log({"confusion": wandb.Image("survival_confusion_matrix.png"),
                   "roc": wandb.Image("survival_roc_curves.png"),
                   "importance": wandb.Image("survival_feature_importance.png")})
        wandb.finish()


if __name__ == "__main__":
    main()
