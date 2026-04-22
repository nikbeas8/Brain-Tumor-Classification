import io

import numpy as np
import tensorflow as tf
from PIL import Image

from backend.config import CLASS_NAMES, IMG_SIZE, MODEL_PATH
from backend.services.gradcam_service import build_grad_cam


model = None


def get_model():
    global model
    if model is None:
        model = tf.keras.models.load_model(MODEL_PATH)
    return model


def prepare_image(file_storage):
    image_bytes = file_storage.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize(IMG_SIZE)
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    return image, tf.cast(image_array, tf.float32)


def predict_mri(file_storage):
    loaded_model = get_model()
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

    return {
        "prediction": predicted_label,
        "confidence": confidence,
        "percentage": round(confidence * 100, 2),
        "grad_cam": grad_cam,
        "probabilities": probabilities,
    }
