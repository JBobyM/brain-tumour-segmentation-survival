# Technical Report — Brain Tumour Segmentation & Survival Prediction

## 1. Introduction

Medical imaging data differs fundamentally from natural images: it is stored in
specialised 3D formats (NIfTI/DICOM), exhibits severe class imbalance (small
lesions in large volumes of healthy tissue), and carries a high bar for
interpretability because clinical decisions depend on the output. This project
builds an end-to-end pipeline that ingests raw multi-modal MRI, segments tumour
sub-regions, predicts a clinical outcome, and explains its predictions.

## 2. Data

| | Segmentation | Survival |
|---|---|---|
| Dataset | MSD Task01_BrainTumour | BraTS 2020 |
| Cases | 484 labelled | 235 with survival labels |
| Modalities | FLAIR, T1w, T1gd, T2w | FLAIR, T1, T1ce, T2 |
| Labels | edema / non-enhancing / enhancing | overall survival (days), age, resection |

The two datasets are the same MRI family, but the Decathlon release **deliberately
re-anonymised filenames to prevent linking cases back to BraTS**, so survival
labels cannot be joined onto the Decathlon volumes. BraTS 2020 (which ships imaging
*and* `survival_info.csv` under shared IDs) is therefore used for the survival stage.

## 3. Segmentation

**Preprocessing.** Load 4-modality NIfTI → channel-first → RAS orientation →
1 mm isotropic resampling → per-channel z-score normalisation over non-zero voxels.
Training uses random 128³ patches with flips and intensity jitter; validation uses
full volumes with sliding-window inference.

**Targets.** The 3 standard overlapping BraTS regions — TC (tumour core), WT (whole
tumour), ET (enhancing tumour) — trained as a **multi-label** problem with sigmoid
activation (the regions are nested, not mutually exclusive).

**Model.** MONAI 3D `UNet`, channels (16, 32, 64, 128, 256), residual units,
instance norm. **Dice Loss** (sigmoid) directly optimises overlap and handles the
severe foreground/background imbalance that would defeat plain cross-entropy.
Trained 50 epochs, Adam + cosine LR, AMP, `DataParallel` across 2× RTX 3090.

**Result.** Best validation **mean Dice 0.713** (TC 0.699 · WT 0.714 · ET 0.732),
still improving at epoch 50.

### 3.1 Engineering finding — a silent label-convention bug

The first training run produced **ET Dice = 0.0000 for all 50 epochs** and WT capped
at 0.52. Root cause: MONAI's built-in `ConvertToMultiChannelBasedOnBratsClassesd`
assumes the *original* BraTS label convention (1 = core, 2 = edema, 4 = ET), but
Decathlon Task01 uses **1 = edema, 2 = non-enhancing, 3 = enhancing** — there is no
label 4. The transform therefore built an all-empty ET channel and a mislocated TC
channel, training the model against scrambled targets.

The fix was a custom `ConvertDecathlonBratsLabelsd` transform (TC = 2∨3, WT = 1∨2∨3,
ET = 3), verified by voxel counts and the nesting invariant WT ⊇ TC ⊇ ET. This lifted
mean Dice **0.488 → 0.713** and ET from **0.0 → 0.73**.
*Takeaway: always inspect `np.unique(labels)` and per-channel voxel counts before
trusting a dataset's label transform.*

## 4. Survival prediction

**Task.** 3-class overall survival, per the BraTS challenge convention:
short (<10 mo / <300 d), mid (10–15 mo / 300–450 d), long (>15 mo / >450 d).
Classes are reasonably balanced (89 / 59 / 86).

**Features.** The Phase-2 U-Net is run end-to-end on each BraTS case to produce
predicted masks, from which interpretable features are derived — per-region volumes
(TC/WT/ET, necrotic core, edema), ratios (enhancing fraction ET/WT, core fraction),
and whole-tumour shape (bounding-box extent, compactness) — plus clinical covariates
(age, resection status). Features are also computed from the **expert masks** as an
upper bound.

**Model.** Random Forest (balanced class weights), evaluated with stratified 5-fold
cross-validation. Metric: accuracy + **macro one-vs-rest ROC-AUC** (per the proposal).

**Results (full cohort, n = 234):**

| Feature set | Accuracy | Macro AUC |
|---|---|---|
| Clinical only (baseline) | 0.406 | 0.556 |
| Predicted masks (end-to-end) | 0.457 | **0.608** |
| Expert masks (upper bound) | 0.500 | **0.650** |

Imaging features add real prognostic signal over the clinical baseline, and the
ordering *clinical < predicted < expert* quantifies how segmentation quality
propagates downstream. Per-class ROC shows the model is strongest on the clinically
critical **short-survivor class (AUC 0.70)**; the mid class sits near chance
(AUC 0.50) — a well-documented difficulty in BraTS survival work. Top features are
age, whole-tumour shape, and enhancement ratios — all clinically plausible.

These absolute numbers are **in line with the published BraTS survival literature**
(even challenge winners land near 0.5–0.6 3-class accuracy). Survival from imaging is
genuinely hard; the value here is a well-characterised, honestly-reported result with
a clear baseline → imaging → upper-bound story.

## 5. Explainability

Grad-CAM was applied to the segmentation U-Net's encoder to visualise which regions
drive the whole-tumour prediction. **Finding:** Grad-CAM — designed for classification
CNNs — produces *diffuse* maps on a 3D segmentation network. Measured attention was not
tumour-focused at any encoder depth (inside/outside-tumour ratio ≤ 1; the deepest
bottleneck actively anti-localised). Using a 128-channel mid-encoder layer with
per-case contrast gives the most usable result: attention concentrates on the
**tumour-bearing hemisphere**, while the predicted segmentation mask provides the
precise boundary. Occlusion sensitivity would be a better-suited method and is a
natural extension.

## 6. Deployment

A Streamlit app (`app.py`) exposes the full pipeline: select a bundled sample case or
upload four modalities → segmentation overlay with a slice slider → survival class and
probabilities (with ground-truth comparison for samples) → Grad-CAM overlay → tumour
volume metrics.

## 7. Limitations & future work

- **Segmentation** trained on Decathlon and applied to BraTS 2020 (domain shift);
  the model slightly over-segments. Fine-tuning on BraTS, larger patches, longer
  training, or SegResNet would raise Dice (WT toward ~0.85).
- **Survival** has small N (235; ~118 GTR-only) and modest AUC, consistent with the
  literature. Deep encoder features, radiomics, or a survival-analysis model (Cox /
  time-to-event) are natural next steps.
- **Explainability** would benefit from occlusion sensitivity or SHAP on the survival
  features (feature importances are already reported).

## 8. Reproducibility

All stages are scripted and seeded where applicable; see `README.md` for exact
commands. Environment pinned in `requirements.txt`; segmentation training tracked in
Weights & Biases.
