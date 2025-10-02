# Sign Language Recognition

A lightweight, script-first project for **Sign Language** recognition using landmark extraction, a trainable classifier, and real-time webcam inference.

This repository contains three main scripts, each with a single responsibility:

- `extract_landmarks.py` — Read images/videos, detect keypoints (e.g., hands/pose/face), and save per-sample landmark arrays to disk.
- `train_classifier.py` — Load saved landmarks, build a train/val split, train a classifier, and export model weights.
- `webcam_infer.py` — Load trained weights and run live predictions from your webcam with optional on-screen overlays.

> **Good to know:** The dataset itself is **not** included in this repo. To reproduce results, users must either download a public dataset or collect their own clips and run `extract_landmarks.py` first. See **Dataset & Expected Layout** below.

---

```bash
# Python 3.9+ is recommended
python -m venv .venv
# Windows
. .venv/Scripts/activate
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
```

---
## File Overview

```
extract_landmarks.py   # Detects landmarks from images/videos and saves .npy tensors
train_classifier.py    # Trains a classifier on saved landmarks and exports weights
webcam_infer.py        # Real-time inference via webcam using exported weights
```

---

## Dependencies

Create a `requirements.txt` like the following (trim to match your code):

```
numpy
pandas
scikit-learn
matplotlib
opencv-python
mediapipe
torch           # or tensorflow
torchvision     # if using torch image ops
tqdm
pydantic        # if you parse configs
pyyaml          # if using YAML configs
```

Then install with `pip install -r requirements.txt`.

---
