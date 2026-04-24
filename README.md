# Brain Tumor Classification Full-Stack AI App

This project is a full-stack AI web application for classifying brain MRI images into four classes:

- Glioma
- Meningioma
- Pituitary
- No Tumor

The backend uses a trained TensorFlow/Keras EfficientNetB0 model. The frontend provides a modern upload interface, prediction results, confidence scores, class probabilities, and Grad-CAM visualization for tumor predictions.

> Disclaimer: This project is for educational and research use only. It is not a substitute for professional medical diagnosis.

## Features

- Flask backend for model inference
- HTML, CSS, and JavaScript frontend
- MRI image upload with preview
- Four-class prediction
- Confidence score and class probabilities
- Grad-CAM heatmap overlay for tumor classes
- No Grad-CAM generated for `notumor`
- Clean backend/frontend project structure
- Production runner support with Waitress

## Project Structure

```text
brain-tumor-classification/
│
├── backend/
│   ├── __init__.py
│   ├── app.py
│   ├── config.py
│   └── services/
│       ├── __init__.py
│       ├── gradcam_service.py
│       └── prediction_service.py
│
├── frontend/
│   ├── templates/
│   │   └── index.html
│   └── static/
│       ├── css/
│       │   └── styles.css
│       └── js/
│           └── script.js
│
├── models/
│   └── best_model.keras
│
├── notebooks/
│   ├── main.ipynb
│   ├── predict.ipynb
│   └── grad_cam.ipynb
│
├── dataset/
│   ├── train/
│   └── test/
│
├── run.py
├── requirements.txt
└── README.md
```

## Tech Stack

- Python
- Flask
- TensorFlow / Keras
- OpenCV
- NumPy
- Pillow
- HTML
- CSS
- JavaScript

## Setup

### 1. Create a virtual environment

```bash
python -m venv .venv
```

### 2. Activate the environment

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the trained model

Make sure the model exists here:

```text
models/best_model.keras
```

The current model file is around 66 MB. GitHub can store it, but GitHub may warn for files over 50 MB. For long-term use, Git LFS is recommended for model files.

## Run Locally

```bash
python run.py
```

Open:

```text
http://127.0.0.1:5000
```

If port `5000` is already busy, run with another port:

Windows PowerShell:

```powershell
$env:PORT="5050"; python run.py
```

macOS/Linux:

```bash
PORT=5050 python run.py
```

## Production Run

For a simple production-style local run:

```bash
waitress-serve --host=0.0.0.0 --port=5000 run:app
```

Then open:

```text
http://127.0.0.1:5000
```

## Deploy on Render

This repository is prepared for Render deployment with:

- `render.yaml`
- `Procfile`
- `.python-version`

Recommended service settings:

- Build command: `pip install -r requirements.txt`
- Start command: `waitress-serve --host=0.0.0.0 --port=$PORT run:app`
- Health check path: `/health`

## Demo Deployment Notes

If you deploy this project now, treat it as a demo only:

- Keep the educational-use disclaimer visible
- Do not present predictions as medical diagnosis
- Expect weaker reliability on unseen real-world MRI images
- Treat Grad-CAM as an illustrative explanation, not proof

## Quick Evaluation

Run this before demo deployment:

```bash
python scripts/evaluate_demo.py
```

This script performs:

- a labeled spot-check on `dataset/test`
- an optional evaluation on files inside `web_samples/`
- a Markdown report export to `reports/demo_evaluation_report.md`

Example with a larger sample:

```bash
python scripts/evaluate_demo.py --limit-per-class 25
```

## API Endpoints

### `GET /`

Serves the web interface.

### `GET /health`

Returns backend health status.

Example response:

```json
{
  "status": "ok"
}
```

### `POST /predict`

Accepts one uploaded MRI image using form field name `image`.

Supported file types:

- JPG
- JPEG
- PNG
- WEBP

Example response:

```json
{
  "prediction": "meningioma",
  "confidence": 1.0,
  "percentage": 100.0,
  "grad_cam": "data:image/png;base64,...",
  "probabilities": [
    {
      "label": "glioma",
      "confidence": 0.0,
      "percentage": 0.0
    }
  ]
}
```

For `notumor` predictions, `grad_cam` is returned as `null`.

## Model Details

- Architecture: EfficientNetB0 transfer learning model
- Input size: `224 x 224 x 3`
- Classes: `glioma`, `meningioma`, `notumor`, `pituitary`
- Saved model path: `models/best_model.keras`

## Grad-CAM Behavior

Grad-CAM is generated only for tumor predictions:

- `glioma`
- `meningioma`
- `pituitary`

Grad-CAM is skipped for:

- `notumor`

This keeps the UI clinically clearer and avoids showing a tumor heatmap when the model predicts no tumor.

## Dataset

Dataset source:

```text
https://www.kaggle.com/datasets/deeppythonist/brain-tumor-mri-dataset
```

Expected dataset location:

```text
dataset/
```

The dataset is used for training and evaluation notebooks. It is not required for running prediction if `models/best_model.keras` already exists.

## Notebooks

The notebooks are kept for experimentation and training:

- `notebooks/main.ipynb`: training workflow
- `notebooks/predict.ipynb`: notebook-based prediction testing
- `notebooks/grad_cam.ipynb`: original Grad-CAM experimentation

The production app logic now lives in `backend/`.
