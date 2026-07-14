# Technical report

The deep dive behind the [README](README.md) — same story, more numbers.

## Data

Two datasets, both multi-modal MRI (FLAIR, T1, T1ce, T2), skull-stripped and co-registered.

- **Segmentation:** Medical Segmentation Decathlon Task01, 484 labelled volumes.
- **Survival:** BraTS 2020, 235 cases with survival labels.

They're the same tumours, but Decathlon re-anonymised its filenames on purpose so you
can't map a case back to BraTS. That's why survival runs on BraTS 2020, which ships the
outcome labels itself.

## Segmentation

Preprocessing: 4-modality NIfTI, RAS orientation, 1 mm resampling, per-channel z-score over
non-zero voxels. Train on random 128³ patches with flips and intensity jitter; validate on
full volumes with sliding-window inference.

The targets are the three overlapping BraTS regions (whole tumour, core, enhancing), trained
multi-label with sigmoid and Dice loss. The regions are nested, not exclusive, so softmax
would be wrong.

I trained two architectures on identical setups and kept the winner:

| Architecture | Epochs | Mean Dice | TC | WT | ET |
|---|---|---|---|---|---|
| U-Net (baseline) | 50 | 0.713 | 0.699 | 0.714 | 0.732 |
| **SegResNet (adopted)** | 100 | **0.834** | 0.820 | **0.889** | 0.797 |

SegResNet won on every region, so everything downstream uses it. Trained with Adam, cosine
LR, and mixed precision on two RTX 3090s. On a 30-volume validation subset: WT Dice 0.90
(sensitivity 0.93), TC 0.86, ET 0.85, specificity ~0.999. About a second per volume, 5.9 GB.

### The label bug

The first run gave enhancing-tumour Dice of exactly 0.0 for all 50 epochs. MONAI's built-in
BraTS label converter expects labels 1/2/4; Decathlon uses 1/2/3. There is no label 4, so
the enhancing channel came out empty and the core channel was mislocated. I trained against
scrambled targets until I printed the per-channel voxel counts. A custom converter (core =
2 or 3, whole = 1/2/3, enhancing = 3) fixed it and took mean Dice from 0.49 to 0.71. Lesson:
run `np.unique` on your labels before trusting any transform.

### Predicted vs ground truth

![Predicted vs ground-truth segmentation](pred_vs_gt_segmentation.png)

*Best, median, and worst case by Dice. Green is the expert outline, red is the model's, and
the error map shows where they agree (green), where the model missed tumour (red), and where
it over-called (orange). Even the worst case (0.80) finds the tumour.*

![Predicted vs expert tumour volume](pred_vs_gt_volume.png)

*Predicted vs expert tumour volume, one dot per patient. The closer to the diagonal, the
better the model measures size. Correlations are 0.98 (core), 0.97 (whole), 0.93 (enhancing),
which is what makes the survival features trustworthy.*

## Survival

3-class survival with the BraTS cutoffs: short (<10 months), mid (10–15), long (>15). The
classes are roughly balanced (89/59/86).

For each case I run the segmentation model, turn the predicted masks into features
(per-region volumes, ratios like enhancing fraction, whole-tumour shape), and add age and
resection status. A random forest classifies them, scored with stratified 5-fold
cross-validation. I compute the same features from the expert masks too, as an upper bound.

| Features | Accuracy | Macro AUC |
|---|---|---|
| Clinical only (baseline) | 0.41 | 0.56 |
| Predicted masks (end-to-end) | 0.44 | **0.62** |
| Expert masks (upper bound) | 0.50 | 0.65 |

Imaging helps: adding it to age-and-surgery lifts macro AUC from 0.56 to 0.62, close to the
expert-mask ceiling of 0.65. The model is best on short survivors (AUC 0.67) and near chance
on the mid class (0.55), which is the known hard case.

One thing to be precise about: the BraTS challenge scores this by accuracy, and the best
entries only reach about 0.62 (2020 winner 61.7%, all-time ceiling ~0.63). My accuracy is
0.44 (0.50 with expert masks) — above the baselines, below the top entries. The 0.62 I quote
is AUC, a different metric; don't confuse the two. Honest result on a hard task, not a
state-of-the-art claim.

## Explainability

I didn't want to ship a heatmap I couldn't defend, so I measured it. Three localisation
scores against the expert masks (N=20): concentration (heat inside the tumour vs its size,
where >1 beats random), pointing game (does the hottest voxel land in the tumour), and
inside-vs-outside heat.

| Method | Concentration | Pointing game | Inside/outside |
|---|---|---|---|
| Grad-CAM | 0.9× | 0% | 0.9× |
| **Occlusion sensitivity** | **6.2×** | **50%** | **8.6×** |

Grad-CAM fails here. It's built for classification, and on a segmentation network the map is
diffuse — worse than random, with its peak never inside the tumour. Occlusion sensitivity
(hide a patch, measure how much the tumour prediction drops) is the right tool: 6× more
concentrated on the tumour, hitting it half the time. I kept the Grad-CAM analysis in the
repo as a documented negative result. The occlusion loop runs on a 2× downsampled volume for
speed (~8 s/case).

## Deployment

A Streamlit app (`app.py`) runs the whole thing: pick a case or upload four modalities,
segment, get the survival class and probabilities, see the occlusion overlay and tumour
volumes, scroll the slices.

## Limitations

- Segmentation trains on Decathlon and runs on BraTS 2020, so there's a domain shift.
  Fine-tuning on BraTS would help.
- Survival has small N (235) and modest AUC, in line with the literature. A proper
  time-to-event model (Cox) or radiomic features are the obvious next steps.
- SHAP on the survival features would extend the explainability to the prognosis side.

## Reproducing

Everything is scripted and seeded where it matters; the commands are in the README. The
environment is pinned in `requirements.txt`, and segmentation training is logged to Weights
& Biases.
