# 🧠 Brain Tumor Classification using Deep Learning with Explainable AI

## Overview

This project focuses on automated brain tumor detection and classification from MRI scans using Deep Learning and Explainable AI (XAI) techniques.

The model classifies MRI images into:

- Glioma Tumor
- Meningioma Tumor
- Pituitary Tumor
- No Tumor

Additionally, Grad-CAM visualization is used to highlight tumor regions, making the model interpretable and trustworthy.

## Dataset
Download dataset from Kaggle:

https://www.kaggle.com/datasets/deeppythonist/brain-tumor-mri-dataset

Place it inside:
```
Brain-Tumor-Classification/dataset/
```

## Problem Statement

Manual detection of brain tumors from MRI scans is:
- Time-consuming
- Prone to human error
- Requires expert radiologists

This project aims to:
- Automate tumor classification
- Improve accuracy
- Provide visual explanations for predictions

## Features
- Multi-class tumor classification
- High accuracy deep learning model
- Transfer Learning (EfficientNetB0)
- Grad-CAM for Explainability
- Optimized for GPU (Google Colab - T4)
- Clean and modular code structure

## Tech Stack
- Python
- TensorFlow / Keras
- OpenCV
- NumPy / Pandas
- Matplotlib / Seaborn
- Grad-CAM (Explainable AI)

## Project Structure
```text
Brain-Tumor-Classification/
│
├── dataset/
│   ├── train/
│   ├── test/
│
├── models/
│   └── best_model.keras
│
├── main.ipynb
├── grad_cam.ipynb
├── predict.ipynb
|
├── requirements.txt
└── README.md
```

## Model Details
- Architecture: EfficientNetB0 (Transfer Learning)
- Input Size: 224 × 224 × 3
- Optimizer: Adam
- Loss Function: Categorical Crossentropy
- Callbacks Used:
    - EarlyStopping
    - ReduceLROnPlateau
    - ModelCheckpoint

## Performance
- Training Accuracy: 98.78%
- Validation Accuracy: 98.25%

## How to Run
### 1️⃣ Clone Repository
```bash
git clone https://github.com/nikbeas8/Brain-Tumor-Classification.git
cd Brain-Tumor-Classification
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
### 3️⃣ Train Model

- Open **main.ipynb** in Google Colab or VS code. (I suggest Google Colab)
- Then run all the cells

### 4️⃣ Test Model

- Open **predict.ipynb** in Google Colab or VS code.
- Then run all the cells

## Grad-CAM (Explainable AI)

Grad-CAM helps visualize where the model is focusing in the MRI image.

### Run Grad-CAM

- Open **grad_cam.ipynb** in Google Colab or VS code.
- Then run all the cells

## Output:
- Heatmap showing tumor region
- Overlay on original MRI image

## Sample Output

<img width="1484" height="767" alt="image" src="https://github.com/user-attachments/assets/7f00f60b-cca6-40ce-8cad-8c70157b6a66" />


## Future Improvements
- Web-based interface (Frontend Integration)
- 3D Brain Visualization
- Real-time hospital deployment

## ⚠️ Disclaimer

This project is for educational and research purposes only.
It should not be used as a substitute for professional medical diagnosis.

## If you like this project

Give it a ⭐ on GitHub — it really helps!
