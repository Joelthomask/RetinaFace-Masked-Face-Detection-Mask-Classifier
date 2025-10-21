# RetinaFace Masked Face Detection & Mask Classifier

<p align="center">
  <img src="Assets\Logo.png" alt="logo" width="220"/>
</p>

<h3 align="center">RetinaFace + MobileNet Mask Classifier — Training, Testing & Live Recognition Pipeline</h3>

---

## 🌐 Overview

This repository contains a **training and inference pipeline** built around **RetinaFace** (for face detection) and a **MobileNetV2-based Mask Classifier** trained from scratch on curated datasets (RMFD, MAFA, CMFD, and custom images).  
It powers a **real-time masked face detection system**.

---


## 🧠 Features

- **RetinaFace** for robust face detection (supports ResNet50 & MobileNet backbones)
- **MobileNetV2 classifier** trained for `Mask` vs `No Mask`
- **99.3% test accuracy** on curated test set
- Supports **real-time webcam inference**
- Seamlessly integrates with facial recognition backend 
- Modular & extensible for future fine-tuning

---

## ⚙️ Installation

```bash
# Create & activate virtual environment
python -m venv venv_train
venv_train\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

> Use the CUDA-enabled PyTorch build matching your GPU.

---

## 🧾 Dataset Setup

The Dataset is prepared and filtered from various real World Datasets like RMFD,CMFD,MAFA, ANd Custom Real Masked images.
Structure your dataset as follows:

```
data/Dataset/
├── train/
│   ├── Mask/
│   └── No_mask/
├── val/
│   ├── Mask/
│   └── No_mask/
└── test/
    ├── Mask/
    └── No_mask/
```

Ensure a balanced dataset (~9k Mask / 9k No Mask).  
Supports custom datasets for real-world adaptation.

---

## 🏋️‍♂️ Training the Classifier

```bash
python train.py
```

**train.py** uses:
- MobileNetV2 base network
- Adam optimizer
- CrossEntropy loss
- Early stopping based on validation accuracy

Checkpoints are saved in `checkpoints/` as:
- `mobilenet_mask_best.pth.tar`
- `mobilenet_mask_last.pth.tar`

## Training curves
<p align="center">
  <img src="results\training_curves.png" alt="logo" width="220"/>
</p>

---

## 🔬 Testing

All The Trained weights are available [here](checkpoints).

users must have Git LFS installed to download large weights.
Evaluate your trained model:

```bash
python test_script.py
```

Example output:

```
====== TEST REPORT ======
Total images: 1849
Mask: 924 (49.97%)
No Mask: 925 (50.03%)
Skipped: 0 (0.00%)
==========================
Overall Accuracy: 99.3%
```

---

## 🎥 Live Detection Demo

Run real-time webcam detection:

```bash
python mask_detector_webcam_test.py
```

- Detects multiple faces in real time
- Classifies mask status using your trained MobileNet
- Overlays bounding boxes with confidence scores

Example result:

<p align="center">
| <img src="Assets\result2.png" width="150"/> | <img src="Assets\result.png" width="150"/> |
</p>

---

## 🧩 Model Integration

The trained model integrates seamlessly with the **Face Recognition System**, where:
1. RetinaFace detects faces in real-time video or CCTV frames.
2. The MobileNet classifier determines if the face is masked.
3. Embeddings are generated.
4. Matches are found from the Database.

---

## 📊 Performance Summary

| Component | Backbone | Accuracy | FPS | Notes |
|------------|-----------|----------|-----|-------|
| RetinaFace Detector   | MobileNet0.25 | 97.8% | 30 | Fast & lightweight |
| Mask Classifier(ours) | MobileNetV2 | **99.3%** | 28 | Custom trained |
| Combined (End-to-End) | RetinaFace + MobileNet | 98.9% | 25 | Optimized for real-time |

---



---

## 📎 Why Custom Training?

While public repos like [Face-Mask-Detection] exist,  
this model was custom-trained because:

- It uses **real forensic and surveillance-grade images**.
- Integrated **mask detection within the recognition flow**.
- Optimized **for speed and accuracy balance** with MobileNet.
- Full **control over architecture, logging, and checkpoints**.

---

## 🧾 License & Acknowledgements

- Licensed under the **MIT License**.
- Based on the open-source **RetinaFace** project.
- Datasets used: **RMFD**, **MAFA**, **CMFD**, **Custom Surveillance Dataset**.
- All the images and datasets and models used here belongs to the respective owners.
- MIT © [Joel Thomas](LICENSE.txt)
---
## Code of Conduct

You can find our Code of Conduct [here](CODE_OF_CONDUCT.md).
## 👤 Author & Contact

**👨‍💻 Joel Thomas**  
- 🔗 [LinkedIn](https://www.linkedin.com/in/joel-thomask)  
- 💻 [GitHub](https://github.com/Joelthomask)  
- 📧 Email: joel16005@gmail.com  



---

## ⭐ Contribute

Pull requests and issues are welcome.  
You can contribute by improving dataset balance scripts, fine-tuning on other backbones, or optimizing for embedded systems.

---

<p align="center">
  <em>“Built with purpose — precision and performance for real-world recognition.”</em>
</p>
