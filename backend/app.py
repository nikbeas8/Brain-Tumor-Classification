from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import RequestEntityTooLarge

from backend.config import (
    ALLOWED_EXTENSIONS,
    FRONTEND_DIR,
    MAX_CONTENT_LENGTH,
    WARM_MODEL_ON_STARTUP,
)
from backend.services import predict_mri, warm_model


def create_app():
    app = Flask(
        __name__,
        template_folder=str(FRONTEND_DIR / "templates"),
        static_folder=str(FRONTEND_DIR / "static"),
        static_url_path="/static",
    )
    app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

    @app.route("/")
    def home():
        return render_template("index.html")

    @app.route("/health")
    def health():
        return jsonify({"status": "ok"})

    @app.route("/predict", methods=["POST"])
    def predict():
        if "image" not in request.files:
            return jsonify({"error": "Please upload an MRI image."}), 400

        image_file = request.files["image"]
        if image_file.filename == "":
            return jsonify({"error": "No file selected."}), 400

        if not allowed_file(image_file.filename):
            return jsonify({"error": "Only JPG, JPEG, PNG, and WEBP images are supported."}), 400

        try:
            return jsonify(predict_mri(image_file))
        except Exception as exc:
            return jsonify({"error": f"Prediction failed: {exc}"}), 500

    @app.errorhandler(RequestEntityTooLarge)
    def handle_file_too_large(_error):
        return jsonify({"error": "Image is too large. Please upload a file under 8 MB."}), 413

    if WARM_MODEL_ON_STARTUP:
        warm_model()

    return app


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
