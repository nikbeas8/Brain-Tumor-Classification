import io
from time import perf_counter

import numpy as np
import tensorflow as tf
from PIL import Image

from backend.config import (
    CLASS_NAMES,
    DATASET_NAME,
    DATASET_SOURCE,
    DATASET_USAGE_NOTE,
    HIGH_CONFIDENCE_THRESHOLD,
    IMG_SIZE,
    MEDIUM_CONFIDENCE_THRESHOLD,
    MODEL_NAME,
    MODEL_PATH,
    MODEL_VERSION,
    TRAINING_METRICS_NOTE,
)
from backend.services.gradcam_service import build_grad_cam


model = None


def get_model():
    global model
    if model is None:
        model = tf.keras.models.load_model(MODEL_PATH)
    return model


def warm_model():
    loaded_model = get_model()
    dummy_input = tf.zeros((1, IMG_SIZE[0], IMG_SIZE[1], 3), dtype=tf.float32)
    loaded_model.predict(dummy_input, verbose=0)


def prepare_image(file_storage):
    image_bytes = file_storage.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize(IMG_SIZE)
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    return image, tf.cast(image_array, tf.float32)


def build_assessment(predicted_label, confidence, probabilities):
    sorted_probabilities = sorted(
        probabilities, key=lambda item: item["confidence"], reverse=True
    )
    runner_up = sorted_probabilities[1] if len(sorted_probabilities) > 1 else None
    margin = confidence - (runner_up["confidence"] if runner_up else 0.0)

    if confidence >= HIGH_CONFIDENCE_THRESHOLD and margin >= 0.35:
        confidence_band = "high"
        summary = (
            f"AI estimate leans toward {predicted_label} on this image, but it still "
            "needs confirmation by a qualified clinician."
        )
    elif confidence >= MEDIUM_CONFIDENCE_THRESHOLD and margin >= 0.15:
        confidence_band = "medium"
        summary = (
            f"AI estimate suggests {predicted_label}, but the result is uncertain and "
            "should be treated cautiously."
        )
    else:
        confidence_band = "low"
        summary = (
            "AI estimate is low-confidence on this image. Do not rely on this result "
            "without expert review."
        )

    return {
        "confidence_band": confidence_band,
        "runner_up": runner_up["label"] if runner_up else None,
        "runner_up_confidence": (
            float(runner_up["confidence"]) if runner_up else None
        ),
        "margin": round(float(margin) * 100, 2),
        "summary": summary,
        "disclaimer": (
            "For educational use only. This output is not a medical diagnosis and "
            "must not be used for clinical decision-making."
        ),
    }


def build_technical_details():
    return {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "input_size": f"{IMG_SIZE[0]} x {IMG_SIZE[1]} x 3",
        "dataset_name": DATASET_NAME,
        "dataset_source": DATASET_SOURCE,
        "dataset_usage_note": DATASET_USAGE_NOTE,
        "training_metrics_note": TRAINING_METRICS_NOTE,
    }


def predict_mri(file_storage):
    loaded_model = get_model()
    started_at = perf_counter()
    image, image_array = prepare_image(file_storage)
    predictions = loaded_model.predict(image_array, verbose=0)[0]
    predicted_index = int(np.argmax(predictions))
    predicted_label = CLASS_NAMES[predicted_index]
    confidence = float(np.max(predictions))

    grad_cam = None
    if predicted_label != "notumor":
        grad_cam = build_grad_cam(loaded_model, image, image_array)

    probabilities = [
        {
            "label": CLASS_NAMES[index],
            "confidence": float(score),
            "percentage": round(float(score) * 100, 2),
        }
        for index, score in enumerate(predictions)
    ]
    assessment = build_assessment(predicted_label, confidence, probabilities)
    inference_ms = round((perf_counter() - started_at) * 1000, 2)

    return {
        "prediction": predicted_label,
        "confidence": confidence,
        "percentage": round(confidence * 100, 2),
        "inference_ms": inference_ms,
        "grad_cam": grad_cam,
        "probabilities": probabilities,
        "assessment": assessment,
        "technical_details": build_technical_details(),
    }
