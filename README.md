# CSCI460 — Programming Project II

## Br35H :: Brain Tumor Detection 2020

**Due:** Dec 7, 2025 — Midnight

---

## Project overview

This repository implements a deep-learning system to classify brain MRI images as **`yes`** (tumor present) or **`no`** (no tumor). The model is built with TensorFlow/Keras and is intended to run directly on the department machine `hopper.winthrop.edu` using the provided TensorFlow virtual environment.

The code reads images from the shared dataset (no local copying), preprocesses them, trains a CNN, and evaluates performance on training, validation and a held-out test set. The deliverables are: code, trained model weights, and a short report (3 pages max) describing design and evaluation.

---

## Data location (DO NOT COPY files)

All images are available on `hopper.winthrop.edu` at the following paths:

* `/data/csci460/BTD/yes` — 1500 images containing tumors
* `/data/csci460/BTD/no`  — 1500 images without tumors

Your code **must** read images directly from these directories (so the instructor can run it from the same paths).

---

## Recommended train/validation/test split

A common choice that balances training data with reliable testing is:

* Training (TR): **70%** (2100 images — 1050 yes, 1050 no)
* Validation (VL): **15%** (450 images — 225 yes, 225 no)
* Test (TT): **15%** (450 images — 225 yes, 225 no)

Justify your split choice in your report. Keep class balance within each split (stratified sampling) and set a random seed for reproducibility.

---

## Environment setup

On `hopper.winthrop.edu` a TensorFlow environment is available:

```bash
# in your project directory on hopper
source /data/shared-venvs/tensorflow-standard/bin/activate

# install project-specific dependencies (run once)
pip install -r requirements.txt
```

A sample `requirements.txt` (include in repo):

```
tensorflow
numpy
pandas
scikit-learn
matplotlib
opencv-python
Pillow
seaborn
```

---

## Files in this repository (suggested)

```
train.py           # main training script
evaluate.py        # evaluate saved model on TR/VL/TT sets, compute per-class metrics
model.py           # CNN model definition(s) and helper to load/save models
data_utils.py      # image loader, preprocessing, train/val/test split (stratified)
config.yaml        # experiment settings (image size, batch, learning rate, seed)
requirements.txt
README.md           # this file
report.pdf         # your 3-page report (submit with code)
```

---

## How to run (examples)

1. Train a model with default config:

```bash
source /data/shared-venvs/tensorflow-standard/bin/activate
python train.py --config config.yaml --data-dir /data/csci460/BTD
```

2. Evaluate a saved model on the holdout test set and produce per-class accuracy + confusion matrix:

```bash
python evaluate.py --model checkpoints/best_model.h5 --data-dir /data/csci460/BTD --split test
```

Scripts must accept the `--data-dir` argument so the instructor can point them to `/data/csci460/BTD`.

---

## Implementation notes & suggestions

* **Image preprocessing:** Resize images to a consistent size (e.g., 224x224 or 128x128). Normalize pixel values to [0,1] or mean-subtract + divide by std.
* **Data augmentation:** Use random rotations, flips, and small zooms during training to improve generalization.
* **Model:** Start with a simple CNN (several Conv2D + MaxPool blocks, followed by dense layers). Optionally evaluate transfer learning (e.g., MobileNetV2 or ResNet50) and compare.
* **Loss & metrics:** Use `binary_crossentropy` (or categorical if you one-hot encode) and report **accuracy**, **precision**, **recall**, **F1**, **confusion matrix**, and **per-class accuracy** (yes and no separately).
* **Early stopping & checkpoints:** Save best model by validation loss or validation F1 and use early stopping.
* **Reproducibility:** Set `numpy` and `tensorflow` seeds and log the seed in your report.

---

## What to include in the 3-page report

* Short introduction and objective
* Data split and justification (include exact counts per split)
* Preprocessing and augmentation choices
* Model architecture and hyperparameters (image size, batch size, learning rate, optimizer, number of epochs)
* Evaluation method: how VL was used, and how TT was held-out. Report per-class accuracy and confusion matrix for TR+VL and TT.
* Results, limitations, and possible future improvements

---

## Submission checklist

* Source code that runs on `hopper.winthrop.edu` and reads images from `/data/csci460/BTD`
* `requirements.txt` or `environment` instructions
* Trained model file(s) (optional but useful) in `checkpoints/`
* `report.pdf` (3 pages max)

---

## Contact / Help

If you need help: reply to the project posting or contact the course staff. Make sure your code accepts `--data-dir` and does not copy the dataset to a local directory when run on the department machine.

Good luck — and remember to justify your design choices in the report!

---

## Linux-specific setup & running instructions

These notes explain how to run the project on a standard Linux workstation or the department machines (e.g., `hopper`) when a prebuilt TensorFlow environment is not provided.

### 1) Create and activate a Python virtual environment (recommended)

```bash
# from your project directory
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

If you need GPU support, make sure the machine has the appropriate NVIDIA drivers and CUDA/cuDNN versions compatible with the TensorFlow release you install. Consult TensorFlow's official installation matrix before installing `tensorflow`.

### 2) Running on Linux (examples)

Train the model (reads data directly from the dataset directory):

```bash
source .venv/bin/activate
python train.py --config config.yaml --data-dir /data/csci460/BTD
```

Evaluate a saved model on the holdout test set:

```bash
source .venv/bin/activate
python evaluate.py --model checkpoints/best_model.h5 --data-dir /data/csci460/BTD --split test
```

All scripts accept `--data-dir` so they
