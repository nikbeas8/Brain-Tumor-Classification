import os
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
FRONTEND_DIR = ROOT_DIR / "frontend"
MODEL_PATH = ROOT_DIR / "models" / "best_model.keras"
MODEL_NAME = "EfficientNetB0"
MODEL_VERSION = "best_model.keras"
DATASET_NAME = "Brain Tumor MRI Dataset"
DATASET_SOURCE = "Kaggle (deeppythonist)"
DATASET_USAGE_NOTE = "Local train/test split used in this project build."
TRAINING_METRICS_NOTE = "Training metrics are documented in the project notebooks."

IMG_SIZE = (224, 224)
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
MAX_CONTENT_LENGTH = 8 * 1024 * 1024
HIGH_CONFIDENCE_THRESHOLD = 0.85
MEDIUM_CONFIDENCE_THRESHOLD = 0.60
WARM_MODEL_ON_STARTUP = os.environ.get("WARM_MODEL_ON_STARTUP", "0") == "1"


def get_port():
    return int(os.environ.get("PORT", 5000))
