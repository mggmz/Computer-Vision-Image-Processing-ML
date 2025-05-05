# 🖼️ Image Keypoint Detection — Branch README

This branch is a **self-contained playground** that implements five classical key-point detectors / descriptors in **Python 3 + OpenCV**.  
The code is meant for quick experimentation and as a stepping-stone toward deep-learning pipelines.

---

## 1 · Contents

| Script | Algorithm | What it does |
|--------|-----------|--------------|
| `sift.py` | **SIFT** – Scale-Invariant Feature Transform | Builds a Gaussian scale-space, finds extrema, assigns orientation and draws rich keypoints on `chihuahua.jpg`. :contentReference[oaicite:0]{index=0}:contentReference[oaicite:1]{index=1} |
| `surf.py` | **SURF** – Speeded-Up Robust Features | Same workflow as SIFT but faster Hessian detector; requires the *contrib* build of OpenCV. :contentReference[oaicite:2]{index=2}:contentReference[oaicite:3]{index=3} |
| `fast.py` | **FAST** – Features from Accelerated Segment Test | Ultra-fast corner detector; no descriptors, just keypoints. :contentReference[oaicite:4]{index=4}:contentReference[oaicite:5]{index=5} |
| `brief.py` | **FAST + BRIEF** | Detects corners with FAST and describes them with 256-bit BRIEF strings. :contentReference[oaicite:6]{index=6}:contentReference[oaicite:7]{index=7} |
| `orb.py` | **ORB** – Oriented FAST & Rotated BRIEF | Adds rotation-invariant FAST plus a learned BRIEF pattern; strong speed/accuracy trade-off. :contentReference[oaicite:8]{index=8}:contentReference[oaicite:9]{index=9} |

> **Why these five?** They illustrate the evolution from heavy, float-vector descriptors (SIFT) to lightweight binary descriptors (BRIEF/ORB) that often seed modern neural approaches.

---

## 2 · Setup

```bash
# 1. (optional) create & activate a virtual environment
python -m venv .venv && source .venv/bin/activate   # Linux / macOS
.\.venv\Scripts\activate                             # Windows

# 2. install core dependencies
pip install opencv-python matplotlib

# 3. install contrib build (needed for SURF & BRIEF)
pip install opencv-contrib-python
````

> *The demo image `chihuahua.jpg` should sit in the same folder as the scripts.*

---

## 3 · Running a script

```bash
python sift.py      # or fast.py | brief.py | orb.py | surf.py
```

Each script will:

1. Load **`chihuahua.jpg`**.
2. Convert it to grayscale.
3. Detect keypoints (plus descriptors where applicable).
4. Draw the keypoints back onto the original image.
5. Display the result via **Matplotlib**.

Swap in a different picture by changing the path in `cv2.imread()`.

---

## 4 · Bridging to Neural Networks

Although these detectors are “classical”, their output can bootstrap deep-learning workflows:

* **Data augmentation** — Generate homography pairs and use the keypoints as pseudo-labels.
* **Hybrid descriptors** — Train a small CNN that replaces BRIEF while keeping FAST corners.
* **Visual SLAM** — Feed ORB landmarks into an LSTM-based pose-graph optimiser.

Feel free to fork the branch and add notebooks or PyTorch / TensorFlow examples.

---

## 5 · Troubleshooting

| Symptom                                                                   | Fix                                                                             |
| ------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| `AttributeError: module 'cv2.xfeatures2d' has no attribute 'SURF_create'` | Confirm you installed **opencv-contrib-python ≥ 4.10**                          |
| Blank Matplotlib window                                                   | Verify the image path or supply an absolute path to `cv2.imread()`              |
| Very few / no keypoints                                                   | Lower the threshold parameters inside the script (e.g. `fast.setThreshold(10)`) |

---

## 6 · License Notice

The branch inherits the root project’s license.
SIFT & SURF patents have expired, but older OpenCV builds may still carry “non-free” flags—verify before commercial use.


Happy flappy!
-Tony 

