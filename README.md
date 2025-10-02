# Sign Language Recognition

A lightweight, script-first project for **Sign Language** recognition using landmark extraction, a trainable classifier, and real-time webcam inference.

This repository contains three main scripts, each with a single responsibility:

- `extract_landmarks.py` — Read images/videos, detect keypoints (e.g., hands/pose/face), and save per-sample landmark arrays to disk.
- `train_classifier.py` — Load saved landmarks, build a train/val split, train a classifier, and export model weights.
- `webcam_infer.py` — Load trained weights and run live predictions from your webcam with optional on-screen overlays.

> **Good to know:** The dataset itself is **not** included in this repo. To reproduce results, users must either download a public dataset or collect their own clips and run `extract_landmarks.py` first. See **Dataset & Expected Layout** below.

---

## Project Status & Reproducibility

Yes — others can run this code **without your dataset**, as long as one of the following is true:

1. You provide **pretrained weights** (e.g., `checkpoints/model.onnx` or `model.pth`).  
   - Then they can run **inference** (`webcam_infer.py`) immediately.
2. You provide **clear instructions** to obtain or collect data, and how to run **feature extraction** (`extract_landmarks.py`) followed by **training** (`train_classifier.py`).

If neither pretrained weights nor dataset instructions are provided, others can still run the scripts, but they won’t be able to **reproduce your trained results**.

---

## Quick Start

### 1) Environment

```bash
# Python 3.9+ is recommended
python -m venv .venv
# Windows
. .venv/Scripts/activate
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
```

> If you don’t have a `requirements.txt` yet, see the **Dependencies** section for a template you can copy.

### 2) Dataset & Expected Layout

Place your dataset under `data/` as **one subfolder per class** (or provide a CSV/JSON index). For example:

```
data/
  A/
    video_001.mp4
    video_002.mp4
  B/
    video_010.mp4
  ...
```

Alternatively, for pre-extracted landmarks:

```
landmarks/
  A/
    sample_001.npy
    sample_002.npy
  B/
    sample_010.npy
```

> You can point `extract_landmarks.py` to `--input_root data/` and `--out_root landmarks/` (see CLI).

### 3) Extract Landmarks

```bash
python extract_landmarks.py   --input_root data/   --out_root landmarks/   --media-type video   --max-frames 32   --hands true --pose true --face false
```

Common flags (adjust to match your implementation):
- `--media-type {video,image}` — input type
- `--max-frames INT` — temporal downsampling/windowing
- `--hands/--pose/--face {true|false}` — which keypoint subsets to save
- `--workers INT` — parallel dataloading
- `--overwrite` — re-generate existing `.npy` files

### 4) Train Classifier

```bash
python train_classifier.py   --landmarks_root landmarks/   --out_dir checkpoints/   --model mlp   --epochs 30   --batch-size 64   --lr 1e-3   --val-split 0.2   --seed 42
```

Expected outputs:
- `checkpoints/last.ckpt` (or `model.pth` / `model.onnx` depending on your code)
- `checkpoints/metrics.json` with accuracy/F1, plus confusion matrix images if enabled

### 5) Webcam Inference

```bash
python webcam_infer.py   --weights checkpoints/last.ckpt   --hands true --pose true --face false   --labelmap labelmap.json   --threshold 0.6   --camera 0
```

Keyboard hints (if implemented):
- `q` to quit, `s` to save a frame, `v` to toggle visualization.

---

## File Overview

```
extract_landmarks.py   # Detects landmarks from images/videos and saves .npy tensors
train_classifier.py    # Trains a classifier on saved landmarks and exports weights
webcam_infer.py        # Real-time inference via webcam using exported weights
```

Optional structure you might adopt:
```
checkpoints/           # Trained weights & logs (not tracked by git if large)
configs/               # YAML/JSON configs
data/                  # Raw media (not in git)
landmarks/             # Preprocessed features
scripts/               # Helper scripts, e.g., dataset splits, eval
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

> GPU/CPU notes: If using **PyTorch**, specify the correct CUDA build in your docs. If using **TensorFlow**, match versions with your Python/CUDA.

---

## Releasing Pretrained Weights (Recommended)

- Export smaller formats (`.onnx`, `.pt`, `.pth`) and keep size < 100 MB when possible.
- Use **GitHub Releases** for large files, or **Git LFS** for files up to a few GB.
- Include a `labelmap.json` mapping indices → class names.

Example `labelmap.json`:
```json
{
  "0": "A",
  "1": "B",
  "2": "C"
}
```

---

## Reproducibility Tips

- Fix seeds: `--seed 42` and `torch.backends.cudnn.deterministic = True` (if PyTorch).  
- Save `pip freeze > requirements-lock.txt`.  
- Log metrics (accuracy, F1, confusion matrix).  
- Document data preprocessing and frame selection.

---

## Troubleshooting

- **No webcam found**: Try `--camera 1` or a different device index.
- **Landmark extraction fails**: Ensure `mediapipe`/OpenCV versions match. Try `--max-frames` smaller or disable `--face` for speed.
- **Class mismatch**: Confirm `labelmap.json` matches training labels and is passed to `webcam_infer.py`.
- **Import errors**: Recreate virtualenv and reinstall requirements.

---

## License

Choose a license that matches your goals (MIT is common for research projects). Add a `LICENSE` file in the repo root.

---

## Citation

If you use external datasets or pre-trained models, please credit them in this section.

---

## Acknowledgements

Thanks to open-source contributors in computer vision and pose/hand tracking libraries.
