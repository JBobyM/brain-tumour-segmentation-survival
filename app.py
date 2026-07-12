"""Phase 5 — Interactive demo: upload/select MRI -> segment -> predict -> explain.

Run with:  .venv/bin/streamlit run app.py
"""

import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

import inference as I

BRATS = Path("data/brats2020/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData")
CLASSES = ["Short (<10 mo)", "Mid (10-15 mo)", "Long (>15 mo)"]

# bundled sample cases (one per survival class) with their ground truth
SAMPLES = {
    "Sample A — short survivor": dict(id="BraTS20_Training_001", age=60.0, gtr=1, days=289, cls=0),
    "Sample B — mid survivor":   dict(id="BraTS20_Training_011", age=61.0, gtr=0, days=421, cls=1),
    "Sample C — long survivor":  dict(id="BraTS20_Training_002", age=52.0, gtr=1, days=616, cls=2),
}

st.set_page_config(page_title="Brain Tumour AI", layout="wide")


@st.cache_resource
def _models():
    return I.get_seg_model(), I.load_survival()


def run_pipeline(paths, age, gtr, known):
    (seg_model, dice), surv = _models()
    img, flair = I.preprocess(paths)
    masks = I.segment(seg_model, img)          # (3,H,W,D) TC,WT,ET
    feat = I.features(masks)
    cls, proba = I.predict_survival(surv, age, gtr, known, feat)
    cam = I.explain(seg_model, img, channel=1)  # occlusion sensitivity, whole-tumour
    return dict(flair=flair, masks=masks, feat=feat, cls=cls, proba=proba, cam=cam, dice=dice)


def _window(b):
    """Radiology-style greyscale window: black background, contrast from brain tissue."""
    v = b[b > 0]
    return (float(np.percentile(v, 1)), float(np.percentile(v, 99))) if v.size else (None, None)


def seg_overlay(base, tc, wt, et):
    b = np.rot90(base); vmin, vmax = _window(b)
    fig, ax = plt.subplots(figsize=(5, 5), facecolor="black")
    ax.imshow(b, cmap="gray", vmin=vmin, vmax=vmax)
    # distinct nested outlines: WT (outer) ⊃ TC ⊃ ET (inner)
    for mask, color in [(wt, "#00ff5f"), (tc, "#ff3b3b"), (et, "#00e5ff")]:
        m = np.rot90(mask)
        if m.any():
            ax.contour(m, levels=[0.5], colors=color, linewidths=1.8)
    ax.axis("off"); ax.set_title("Segmentation outlines (WT green · TC red · ET cyan)", color="white")
    fig.tight_layout(); return fig


def cam_overlay(base, cam_slice):
    b = np.rot90(base); h = np.rot90(cam_slice)
    vmin, vmax = _window(b)
    brain = b != 0
    hv = h[brain]
    hmin, hmax = (float(hv.min()), np.percentile(hv, 99.5)) if hv.size else (0, 1)
    fig, ax = plt.subplots(figsize=(5, 5), facecolor="black")
    ax.imshow(b, cmap="gray", vmin=vmin, vmax=vmax)
    ax.imshow(np.ma.masked_where(~brain, h), cmap="jet", alpha=0.5, vmin=hmin, vmax=hmax)
    ax.axis("off"); ax.set_title("Occlusion sensitivity (whole-tumour)", color="white")
    fig.tight_layout(); return fig


# ---------------- Sidebar: input selection ----------------
st.sidebar.title("🧠 Brain Tumour AI")
st.sidebar.caption("Segmentation · Survival · Explainability")
mode = st.sidebar.radio("Input", ["Bundled sample", "Upload your own"])

paths, age, gtr, known, truth = None, 60.0, 1, 1, None
if mode == "Bundled sample":
    key = st.sidebar.selectbox("Case", list(SAMPLES))
    s = SAMPLES[key]
    paths = [BRATS / s["id"] / f"{s['id']}_{m}.nii" for m in I.MODALITIES]
    age, gtr, known, truth = s["age"], s["gtr"], 1, s
    st.sidebar.info(f"Age {s['age']:.0f} · {'GTR' if s['gtr'] else 'non-GTR'}")
else:
    st.sidebar.write("Upload the 4 MRI modalities (.nii/.nii.gz):")
    ups = {m: st.sidebar.file_uploader(m.upper(), type=["nii", "gz"], key=m) for m in I.MODALITIES}
    age = st.sidebar.number_input("Patient age", 18.0, 95.0, 60.0)
    res = st.sidebar.selectbox("Resection", ["GTR (gross total)", "STR / other", "Unknown"])
    gtr = 1 if res.startswith("GTR") else 0
    known = 0 if res == "Unknown" else 1
    if all(ups.values()):
        tmp = Path(tempfile.mkdtemp())
        paths = []
        for m, f in ups.items():
            p = tmp / f"{m}.nii"
            p.write_bytes(f.getbuffer())
            paths.append(p)

# ---------------- Main ----------------
st.title("Brain Tumour Segmentation & Survival Prediction")
st.caption("3D SegResNet segmentation → tumour features → survival stratification → occlusion-sensitivity explanation")

if paths is None:
    st.warning("Upload all four modalities (FLAIR, T1, T1ce, T2) to run the pipeline.")
    st.stop()

if st.button("▶ Run analysis", type="primary"):
    with st.spinner("Segmenting, predicting and explaining… (a few seconds on GPU)"):
        st.session_state["res"] = run_pipeline(paths, age, gtr, known)
        st.session_state["truth"] = truth

if "res" not in st.session_state:
    st.info("Click **Run analysis** to process the selected case.")
    st.stop()

r = st.session_state["res"]
tc, wt, et = r["masks"]
H, W, D = wt.shape
zmax = int(wt.sum((0, 1)).argmax()) if wt.sum() else D // 2
z = st.slider("Axial slice", 0, D - 1, zmax)

c1, c2 = st.columns(2)
c1.pyplot(seg_overlay(r["flair"][:, :, z], tc[:, :, z], wt[:, :, z], et[:, :, z]))
c2.pyplot(cam_overlay(r["flair"][:, :, z], r["cam"][:, :, z]))

# survival prediction
st.subheader("Predicted survival stratification")
cls, proba = r["cls"], r["proba"]
pc1, pc2 = st.columns([1, 2])
pc1.metric("Predicted class", CLASSES[cls])
truth = st.session_state.get("truth")
if truth is not None:
    ok = "✅" if truth["cls"] == cls else "⚠️"
    pc1.caption(f"{ok} Ground truth: {CLASSES[truth['cls']]} ({truth['days']} days)")
fig, ax = plt.subplots(figsize=(6, 2))
ax.barh(CLASSES, proba, color=["#d62728", "#ff7f0e", "#2ca02c"])
ax.set_xlim(0, 1); ax.set_xlabel("probability")
for i, p in enumerate(proba):
    ax.text(p + 0.01, i, f"{p:.2f}", va="center")
fig.tight_layout(); pc2.pyplot(fig)

# tumour metrics
st.subheader("Tumour metrics (from predicted segmentation)")
f = r["feat"]
m1, m2, m3, m4 = st.columns(4)
m1.metric("Whole tumour", f"{f['pred_vol_wt']/1000:.1f} cm³")
m2.metric("Tumour core", f"{f['pred_vol_tc']/1000:.1f} cm³")
m3.metric("Enhancing", f"{f['pred_vol_et']/1000:.1f} cm³")
m4.metric("Enhancing / WT", f"{f['pred_ratio_et_wt']:.2f}")
st.caption(f"Segmentation model validation Dice: {r['dice']:.3f}  ·  "
           "Survival model: RandomForest (macro OVR-AUC ≈ 0.61, 3-class)")
