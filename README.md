# Student Engagement Recognition Using Deep Learning

### A Comparative Study of Frame-Level and Temporal Models on DAiSEE

**Computer Vision & Deep Learning — University of Verona, A.Y. 2025–26**

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?logo=streamlit)](https://deeplearning-and-computer-vision-project-jb2bnwzoqz98jqm8gqpu7.streamlit.app/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rafay-840/DeepLearning-and-Computer-Vision-Project/blob/main/Student_Engagement_Recognition.ipynb)

This repository contains the complete implementation, experiments, and report for a course project investigating whether deep convolutional neural networks — with and without explicit temporal modeling — can recognize student engagement from facial video. We compare frame-level CNN classifiers against CNN+LSTM temporal models on a subject-disjoint split of the [DAiSEE](https://people.iith.ac.in/vineethnb/resources/daisee/index.html) dataset, with two follow-up experiments testing whether the results generalize to a larger dataset scale and to a 3-class label formulation.

---

## Live Demo

**[Try the live webcam demo →](https://deeplearning-and-computer-vision-project-jb2bnwzoqz98jqm8gqpu7.streamlit.app/)**

The deployed app runs the project's best-performing model (ResNet-18 + LSTM, test macro-F1 = 0.595) live on your webcam via `streamlit-webrtc`. Frames are buffered into 8-frame sequences and classified every 8 frames. All inference runs locally in the app's backend session — no video is stored or transmitted anywhere.

A second mode in the sidebar lets you switch to the 3-class extension (Engaged / Bored / Other); see the [Results](#results-summary) section below for why that mode performs noticeably worse on the minority classes.

---

## Project Summary

| | |
|---|---|
| **Task** | Binary (and 3-class) student engagement classification from facial video |
| **Dataset** | [DAiSEE](https://people.iith.ac.in/vineethnb/resources/daisee/index.html) — 9,068 video clips, 112 participants |
| **Models compared** | MobileNetV3-Large (frame-level), MobileNetV3+LSTM, ResNet-18+LSTM, ResNet-18+LSTM (frozen backbone) |
| **Best result** | ResNet-18 + LSTM — test macro-F1 **0.595** (subject-disjoint test set) |
| **Key finding** | Temporal modeling helps, but all models exhibit majority-class-dominant overfitting that persists across data scale and label formulation — see [Results](#results-summary) |

---

## Why DAiSEE, Not the Originally Proposed Kaggle Dataset

This project originally proposed a Kaggle-hosted Student Engagement Dataset. During development, a perceptual-hash audit revealed **severe session-level data leakage**: most images within each class folder originated from a single continuous recording session (e.g., the "Focused" and "Looking Away" classes each consisted entirely of frames from one session). A model trained on a random split of this data simply memorizes session-specific visual characteristics rather than learning generalizable engagement cues — which is exactly what produced an initially observed but **invalid** 97–98% accuracy.

We diagnosed this, documented it, and migrated the entire pipeline to DAiSEE, using its **official, subject-disjoint train/validation/test split** directly rather than constructing our own. This pivot — and the leakage-detection methodology behind it — is documented in full in the report (Section 4.1) and is itself one of the project's findings.

---

## Repository Structure

```
.
├── Student_Engagement_Recognition.ipynb   # Complete pipeline: data prep, training, evaluation, Grad-CAM
├── streamlit_app/
│   ├── app.py                              # Live webcam demo (binary + 3-class modes)
│   └── requirements.txt
├── report/
│   └── Student_Engagement_Recognition_Report.docx
└── README.md
```

**Not included in this repository** (too large for Git): raw DAiSEE video files, extracted HDF5 frame-sequence files, and trained model checkpoints. These live on Google Drive during development; see [Reproducing the Results](#reproducing-the-results) for how to regenerate them, or [Using a Trained Checkpoint](#using-a-trained-checkpoint) if you just want to run the demo.

### Drive Folder (Datasets, Checkpoints, Logs)

**[Access the full project folder on Google Drive →](https://drive.google.com/drive/folders/1zeY1SWvQjWi53HYl10NjjpBu8yRPW4hC?usp=sharing)**

This folder contains everything too large for Git, organized as:

```
DLCV_Project/
├── checkpoints/          # Trained model weights (.pth) for all 7 model variants
├── logs/                 # Per-epoch training logs (.csv) for every experiment
├── data/                 # Raw DAiSEE.zip (if access was granted to you separately)
├── daisee_*_sequences.h5         # Extracted frame sequences, stratified subset
├── daisee_*_full_sequences.h5    # Extracted frame sequences, full dataset
├── daisee_*_3class_sequences.h5  # Relabeled sequences, 3-class extension
├── daisee_*_manifest.csv         # Full official-split manifests
├── daisee_*_subset.csv           # Stratified subsample manifests
├── daisee_*_3class.csv           # 3-class label manifests
├── final_test_results_*.csv      # Test-set evaluation results per experiment
└── figures/                      # Generated report figures (PNG)
```

If you want to skip retraining entirely, download the relevant checkpoint(s) and logs from here directly — the notebook's final section ("Fetch Saved Results") is designed to read from exactly this structure.

---

## Results Summary

| Model | Dataset | Test Macro-F1 |
|---|---|---|
| MobileNetV3 (frame-level) | Stratified subset | 0.4935 |
| MobileNetV3 + LSTM | Stratified subset | 0.5505 |
| **ResNet-18 + LSTM** | **Stratified subset** | **0.5946** |
| ResNet-18 + LSTM (frozen backbone) | Stratified subset | 0.5485 |
| MobileNetV3 (frame-level) | Full dataset | 0.4674 |
| MobileNetV3 + LSTM | Full dataset | 0.5030 |
| ResNet-18 + LSTM | Full dataset | 0.5279 |
| ResNet-18 + LSTM (uncapped weighting) | Full dataset | 0.5299 |
| ResNet-18 + LSTM (3-class) | Stratified subset | 0.3241 |

Five experiments, summarized:

1. **CNN+LSTM consistently outperforms frame-level classification** (macro-F1 improvement of 0.06–0.10), supporting the use of temporal context for engagement recognition.
2. **Grad-CAM analysis reveals the model attends to background and clothing regions**, not facial features — direct visual evidence that the model is exploiting subject/session-identity cues rather than learning genuine engagement-related behavior.
3. **Scaling to the full official training set did not resolve this** — every model performed *worse*, with disengaged-class recall collapsing to 4.6–6.0%, driven by a class-weighting cap that was too weak against the dataset's true 19.3:1 imbalance.
4. **Removing the weighting cap entirely is necessary but not sufficient**: disengaged recall more than doubled (6.0% → 13.5%), but macro-F1 barely moved, since precision fell correspondingly — confirming the deeper bottleneck is representational (per finding #2), not just loss-function calibration.
5. **A 3-class extension (Engaged / Bored / Other) reproduces the same pattern** at near-chance-level performance (macro-F1 0.324), ruling out the binary framing itself as the cause.

Full discussion, all confusion matrices, and the complete experimental protocol are in the [report](report/Student_Engagement_Recognition_Report.docx).

---

## Reproducing the Results

The full pipeline is implemented in `Student_Engagement_Recognition.ipynb`, designed to run on **Google Colab** with a GPU runtime (A100 recommended given DAiSEE's size).

1. Open the notebook in Colab (badge above), or upload it manually.
2. Obtain DAiSEE by requesting access at the [official dataset page](https://people.iith.ac.in/vineethnb/resources/daisee/index.html) and place `DAiSEE.zip` in your Google Drive under `DLCV_Project/data/`.
3. Run the notebook top to bottom. It will:
   - Extract DAiSEE and build subject-disjoint manifests
   - Construct a stratified subsample (preserving all minority-class clips)
   - Extract 8-frame sequences into compressed HDF5 files
   - Train and evaluate all four core models, run Grad-CAM, and run the two follow-up experiments (full-dataset scaling, 3-class extension)
4. **If you've already run the notebook once:** the final section ("Fetch Saved Results") reloads your saved training logs and test results from Drive and regenerates the report figures — no retraining needed.

Expect the full run to take several hours on an A100, dominated by the full-dataset and uncapped-weighting experiments (Sections 6–7 of the notebook).

---

## Running the Streamlit Demo Locally

```bash
cd streamlit_app
pip install -r requirements.txt
streamlit run app.py
```

Download a trained checkpoint (`resnet18_lstm_best.pth` for binary mode, `resnet18_lstm_3class_best.pth` for the 3-class mode) and place it alongside `app.py`, or update the checkpoint path in the sidebar. See [Using a Trained Checkpoint](#using-a-trained-checkpoint) below.

The app uses `streamlit-webrtc` for continuous webcam streaming — this requires a real browser session with camera permissions and will not run inside Colab.

---

## Using a Trained Checkpoint

Trained checkpoints are not included in this repository due to size, but are available in the [shared Drive folder](https://drive.google.com/drive/folders/1zeY1SWvQjWi53HYl10NjjpBu8yRPW4hC?usp=sharing) under `checkpoints/`, including:

- `resnet18_lstm_best.pth` — the best-performing model (binary, recommended for the demo)
- `resnet18_lstm_3class_best.pth` — the 3-class extension
- `mobilenetv3_lstm_best.pth`, `mobilenetv3_framelevel_best.pth`, `resnet18_lstm_frozen_best.pth` — the remaining ablation models from the report

Download the relevant `.pth` file and place it in `streamlit_app/` (or wherever you run `app.py` from).

---

## Methodology Highlights

A few implementation details worth flagging for anyone reading the code closely, all documented in more depth in the notebook and report:

- **Sequence-consistent augmentation**: horizontal flips and color jitter are sampled once per 8-frame sequence and applied identically to every frame, rather than independently per frame — applying independent augmentation would flip a face's orientation mid-clip, which is physically meaningless and would corrupt the temporal signal.
- **Grad-CAM on a CNN+LSTM model**: standard Grad-CAM assumes a single image and a single class score. We compute a separate heatmap per frame by backpropagating the sequence-level class score through the LSTM to each frame's CNN feature map independently. This also required disabling cuDNN temporarily, since cuDNN's optimized LSTM kernel restricts backward passes to training mode.
- **Subject-disjoint verification**: every data split in this project — the original Kaggle attempt, DAiSEE's stratified subsample, and the 3-class extension — is explicitly checked for zero participant overlap across train/validation/test, rather than assumed.
- **Tie-resolution for the 3-class labels**: DAiSEE's four raw ordinal annotation dimensions (Boredom, Engagement, Confusion, Frustration) tie at the maximum score for 20–30% of clips. Clips with 3-way/4-way ties (no genuine differentiated signal) are dropped; the common Boredom/Engagement 2-way tie is resolved using the already-validated binary engagement label, not an arbitrary default.

---

## Tech Stack

- **PyTorch** / **torchvision** — model architecture and training
- **h5py** — compact storage for extracted frame sequences
- **OpenCV** — video frame extraction and Grad-CAM heatmap overlay
- **scikit-learn** — evaluation metrics (macro-F1, confusion matrices)
- **Streamlit** + **streamlit-webrtc** — live webcam demo
- **Google Colab** (A100 GPU) — training environment

---

## Report

The full written report (16 pages, including 8 figures) is available in [`report/Student_Engagement_Recognition_Report.docx`](report/Student_Engagement_Recognition_Report.docx), covering: Motivation, State of the Art, Objectives, Methodology, Experiments and Results, Discussion, Risk Assessment, Conclusions, and References.

---

## Authors

Rafay Saif (VR546150)
Fayyaz Hussain Shah(VR546175)
MSc Artificial Intelligence, University of Verona

---

## Acknowledgments

- [DAiSEE](https://people.iith.ac.in/vineethnb/resources/daisee/index.html): Gupta, A., D'Cunha, A., Awasthi, K., & Balasubramanian, V. (2016). *DAiSEE: Towards User Engagement Recognition in the Wild.*
- Grad-CAM: Selvaraju, R. R., et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.*
- Course: Computer Vision & Deep Learning, Prof. Vittorio Murino & Prof. Francesco Dibitonto, University of Verona.
