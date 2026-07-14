# Brain tumour segmentation and survival prediction from MRI

A pipeline that reads a brain MRI, finds the tumour, measures it, and estimates the
patient's survival window, then shows which part of the scan drove that call. Built with
PyTorch and MONAI.

![Occlusion-sensitivity overlays](occlusion_overlays.png)

## What it does

Feed it the four MRI sequences a radiologist uses (FLAIR, T1, T1ce, T2) and it:

1. Segments the tumour into its three regions (whole tumour, core, enhancing) with a 3D SegResNet.
2. Turns the masks into measurements: volumes, shape, how much of the tumour is actively growing.
3. Passes those, plus the patient's age and surgery status, to a random forest that sorts
   the case into a short / mid / long survival group.
4. Draws an occlusion map over the scan so you can see where the segmentation model was
   actually looking.

A Streamlit app ties it together: pick a case, run it, scroll through the slices.

## Results, honestly

The segmentation is solid. On held-out validation, whole-tumour Dice is 0.90, core 0.86,
and enhancing 0.85, with sensitivity above 0.83 everywhere, so it rarely misses real
tumour. It runs in about a second per volume and fits in 6 GB, so you don't need a special
machine.

Survival is the hard part, and I won't dress it up. My 3-class accuracy is 0.44 (0.50 if I
hand it the expert masks instead of its own). That beats random (0.33) and always guessing
the biggest class (0.38), but it sits below the best BraTS challenge entries, which top out
near 0.62 accuracy. Predicting survival from a scan is just hard. What I can show is that
the imaging adds real signal on top of age and surgery alone (AUC 0.56 to 0.62), and the
model is best at flagging the short-survivor cases, which are the ones you most want to catch.

## The parts I'm actually proud of

The models were the easy bit. The work that mattered was catching things that would have
quietly ruined the results.

A label bug cost me a full training run. MONAI's built-in BraTS label converter expects the
labels 1/2/4, but the Decathlon dataset uses 1/2/3. So the enhancing-tumour channel came out
empty and I trained against broken targets for 50 epochs before I thought to print the
per-channel voxel counts. Fixing it took mean Dice from 0.49 to 0.71.

I picked the architecture by measuring, not by preference. U-Net and SegResNet, same data,
same everything. SegResNet won on every region (whole-tumour Dice went 0.71 to 0.89), so I
switched and re-ran the rest of the pipeline on it.

And I didn't trust my own explanations. Grad-CAM gave me nice-looking heatmaps, so I checked
whether they actually landed on the tumour. They didn't; the hottest spot was inside the
tumour 0% of the time. I replaced it with occlusion sensitivity and scored it the same way:
it hits the tumour about half the time and concentrates there roughly 6x more than chance.

The full write-up, with every number and the limitations, is in [REPORT.md](REPORT.md).

## Running it

You'll need a CUDA GPU. Set up the environment:

```bash
python3 -m venv .venv
.venv/bin/pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
.venv/bin/pip install -r requirements.txt
```

The `cu128` torch build matters. The default PyPI wheel targets CUDA 13 and will report no
GPU on a 12.8 driver.

Then grab the data and walk the pipeline:

```bash
# datasets (~7 GB + ~4.5 GB)
.venv/bin/hf download Novel-BioMedAI/Medical_Segmentation_Decathlon Task01_BrainTumour.tar \
    --repo-type dataset --local-dir data/ && tar -xf data/Task01_BrainTumour.tar -C data/
.venv/bin/kaggle datasets download awsaf49/brats20-dataset-training-validation -p data/brats2020 --unzip

.venv/bin/jupyter notebook eda.ipynb          # start here: look at the data
.venv/bin/python train.py --arch segresnet --epochs 100 --cache-rate 0.3
.venv/bin/python extract_features.py
.venv/bin/python train_survival.py && .venv/bin/python save_survival_model.py
.venv/bin/streamlit run app.py                # the demo
```

Running on a remote box? [ACCESS.md](ACCESS.md) covers the SSH tunnel for the app and notebook.

## What's in here

| File | What it does |
|---|---|
| `eda.ipynb` | Data exploration: modalities, class imbalance, what relates to survival |
| `data_pipeline.py` | Preprocessing and loaders (plus the label fix) |
| `seg_model.py`, `train.py` | Segmentation model and training (`--arch unet` or `segresnet`) |
| `extract_features.py` | Turns masks into tumour features |
| `train_survival.py`, `save_survival_model.py` | The survival classifier |
| `occlusion.py`, `gradcam.py` | The explanation that works, and the one that didn't |
| `measure_metrics.py`, `measure_gradcam.py` | The benchmarks behind the numbers above |
| `inference.py`, `app.py` | Shared inference code and the Streamlit app |

## Data and credits

Segmentation trains on the [Medical Segmentation Decathlon](http://medicaldecathlon.com/)
(Task01). Survival uses [BraTS 2020](https://www.med.upenn.edu/cbica/brats2020/), the only
one of the two with outcome labels. Both are multi-modal MRI, skull-stripped and
co-registered. The datasets belong to their providers (MSD, BraTS/CBICA); the code here is
for research and portfolio use.
