import base64
import io

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

from backend.config import IMG_SIZE


def image_to_data_url(image):
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def normalize_heatmap(heatmap):
    heatmap = np.nan_to_num(heatmap, nan=0.0, posinf=0.0, neginf=0.0)
    max_value = float(np.max(heatmap)) if heatmap.size else 0.0
    if max_value <= 0.0:
        return np.zeros_like(heatmap, dtype="float32")
    return (heatmap / max_value).astype("float32")


def build_grad_cam(model, image, image_array, intensity=0.6):
    base_model = model.get_layer("efficientnetb0")
    target_layer = base_model.get_layer("top_activation")

    preprocessed_input = tf.keras.applications.efficientnet.preprocess_input(
        tf.identity(image_array)
    )

    base_grad_model = tf.keras.Model(
        inputs=base_model.inputs,
        outputs=[target_layer.output, base_model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, base_output = base_grad_model(preprocessed_input, training=False)
        x = model.get_layer("global_average_pooling2d")(base_output)
        x = model.get_layer("batch_normalization")(x)
        x = model.get_layer("dropout")(x, training=False)
        predictions = model.get_layer("dense")(x)
        predicted_index = tf.argmax(predictions[0])
        class_channel = predictions[:, predicted_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0).numpy().astype("float32")
    heatmap = normalize_heatmap(heatmap)

    heatmap_resized = cv2.resize(heatmap, IMG_SIZE)
    heatmap_resized = cv2.GaussianBlur(heatmap_resized, (7, 7), 0)
    heatmap_resized = np.clip(
        np.nan_to_num(heatmap_resized, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0
    )
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    original = np.array(image).astype("uint8")
    overlay = cv2.addWeighted(original, 1 - intensity, heatmap_color, intensity, 0)
    return image_to_data_url(overlay)
