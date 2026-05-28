# Chest X-ray Classification using Deep Learning

A Deep Learning-based medical image classification project that detects **Pneumonia** from Chest X-ray images using:

* Custom CNN (built from scratch)
* Transfer Learning with MobileNetV2

The project also includes **Grad-CAM visualization** to interpret which regions of the X-ray influenced the model prediction.

---
DATASET :: https://drive.google.com/file/d/1Lx47Vuqf2OXzGeDGAZMfno0TY8RQJB0M/view
# Project Overview

Pneumonia is a serious lung infection that can be identified using Chest X-ray scans.
This project builds an AI model capable of classifying X-ray images into:

* Normal
* Pneumonia

Two different experiments were performed:

1. Simple CNN Model
2. Pretrained MobileNetV2 Model

---

# Features

* Chest X-ray Classification
* Binary Classification (Normal vs Pneumonia)
* Custom CNN Architecture
* Transfer Learning
* Grad-CAM Heatmap Visualization
* Model Saving & Loading using PKL/PTH
* Medical Image Processing

---

# Technologies Used

## Language

* Python

## Frameworks & Libraries

* PyTorch
* Torchvision
* NumPy
* OpenCV
* Matplotlib
* PIL
* pytorch-grad-cam

---

# Dataset Structure

```text
dataset/
│
├── train/
│   ├── normal/
│   └── pneumonia/
│
├── test/
│   ├── normal/
│   └── pneumonia/
```

---

# Dataset Information

| Category  | Training | Testing |
| --------- | -------- | ------- |
| Normal    | 314      | 80      |
| Pneumonia | 326      | 80      |

Total:

* Training Images: 640
* Testing Images: 160

---

# Experiment 1 — Simple CNN

A custom Convolutional Neural Network was built from scratch using:

* Conv2D Layers
* ReLU Activation
* MaxPooling
* Fully Connected Layers

## CNN Architecture

```python
nn.Sequential(

    nn.Conv2d(1,32,3,padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2,2),

    nn.Conv2d(32,64,3,padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2,2),

    nn.Conv2d(64,128,3,padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2,2),

    nn.Flatten(),

    nn.Linear(32768,128),
    nn.ReLU(),

    nn.Linear(128,2)
)
```

---

# Experiment 2 — Transfer Learning

Used pretrained MobileNetV2 model trained on ImageNet.

## Steps:

* Loaded pretrained weights
* Froze feature extraction layers
* Replaced classifier layer
* Fine-tuned on Chest X-ray dataset

---

# Image Preprocessing

```python
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])
```

---

# Why These Preprocessing Steps?

## Grayscale to RGB

MobileNetV2 expects 3-channel input images.

## Resize

All images resized to fixed dimensions for stable training.

## Normalization

Improves convergence and stabilizes training.

---

# Training Configuration

| Parameter     | Value            |
| ------------- | ---------------- |
| Optimizer     | Adam             |
| Loss Function | CrossEntropyLoss |
| Epochs        | 10               |
| Batch Size    | 32               |
| Learning Rate | 0.001            |

---

# Results

| Experiment  | Accuracy         |
| ----------- | ---------------- |
| Simple CNN  | 96.88            |
| MobileNetV2 | 91.88%           |

Observation:

* Transfer Learning achieved higher accuracy
* Faster convergence
* Better feature extraction

---

# Grad-CAM Visualization

Grad-CAM was implemented to visualize:

* Which lung regions influenced predictions
* How the model interprets X-ray images

This improves:

* Explainability
* Transparency
* Medical trustworthiness

---

Model Saving
Save Model using PKL
import pickle

with open("cnn_model.pkl", "wb") as f:
    pickle.dump(model, f)
Load Saved Model
import pickle

with open("cnn_model.pkl", "rb") as f:
    model = pickle.load(f)

model.eval()

The trained model is stored in .pkl format to avoid retraining the model every time during prediction or Grad-CAM visualization.
---

# Future Improvements

* Data Augmentation
* DenseNet Implementation
* Validation Split
* Confusion Matrix
* F1 Score Evaluation
* Web Deployment using Flask/Streamlit

---

# Project Structure

```text
project/
│
├── dataset/
├── outputs/
├── SIMPLE CNN.ipynb
├── Pretrained MobileNet.ipynb
├── README.md
├── experiments.md
├── requirements.txt
└── sample_outputs/
```

---

# Skills Demonstrated

* Deep Learning
* CNN Architecture
* Transfer Learning
* Medical Image Processing
* Explainable AI (XAI)
* Computer Vision
* PyTorch Development

---

# Conclusion

This project demonstrates how Deep Learning can assist in medical image diagnosis using Chest X-ray classification.

Among both experiments, Transfer Learning using MobileNetV2 achieved the best performance while maintaining computational efficiency and interpretability through Grad-CAM visualization.
