const dropArea = document.getElementById("dropArea");
const fileInput = document.getElementById("fileInput");
const fileName = document.getElementById("fileName");
const previewImage = document.getElementById("previewImage");
const uploadContent = document.getElementById("uploadContent");
const analyzeBtn = document.getElementById("analyzeBtn");

const predictionText = document.getElementById("predictionText");
const confidenceText = document.getElementById("confidenceText");

const gliomaValue = document.getElementById("gliomaValue");
const meningiomaValue = document.getElementById("meningiomaValue");
const pituitaryValue = document.getElementById("pituitaryValue");
const noTumorValue = document.getElementById("noTumorValue");

const gliomaBar = document.getElementById("gliomaBar");
const meningiomaBar = document.getElementById("meningiomaBar");
const pituitaryBar = document.getElementById("pituitaryBar");
const noTumorBar = document.getElementById("noTumorBar");

const gradcamImage = document.getElementById("gradcamImage");
const gradcamCard = document.querySelector(".gradcam-card");
const gradcamMessage = document.getElementById("gradcamMessage");

let selectedFile = null;
const API_BASE_URL = window.location.protocol === "file:" ? "http://127.0.0.1:5000" : "";

dropArea.addEventListener("click", () => fileInput.click());

fileInput.addEventListener("change", (event) => {
  if (event.target.files.length > 0) {
    handleFile(event.target.files[0]);
  }
});

["dragenter", "dragover"].forEach((eventName) => {
  dropArea.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropArea.classList.add("is-active");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  dropArea.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropArea.classList.remove("is-active");
  });
});

dropArea.addEventListener("drop", (event) => {
  if (event.dataTransfer.files.length > 0) {
    handleFile(event.dataTransfer.files[0]);
  }
});

function handleFile(file) {
  if (!file.type.startsWith("image/")) {
    showError("Please select a valid image file.");
    return;
  }

  selectedFile = file;
  fileName.value = file.name;
  analyzeBtn.disabled = false;

  const reader = new FileReader();
  reader.onload = (event) => {
    previewImage.src = event.target.result;
    previewImage.classList.remove("hidden");
    uploadContent.classList.add("hidden");
  };
  reader.readAsDataURL(file);

  resetResults();
}

function resetResults() {
  predictionText.textContent = "Waiting for analysis...";
  predictionText.classList.remove("error-text");
  confidenceText.textContent = "0%";

  updateProbability(gliomaValue, gliomaBar, 0);
  updateProbability(meningiomaValue, meningiomaBar, 0);
  updateProbability(pituitaryValue, pituitaryBar, 0);
  updateProbability(noTumorValue, noTumorBar, 0);

  gradcamImage.removeAttribute("src");
  gradcamCard.classList.remove("has-image");
  gradcamMessage.textContent = "Tumor heatmap appears here after analysis.";
}

function showError(message) {
  predictionText.textContent = message;
  predictionText.classList.add("error-text");
  confidenceText.textContent = "0%";
}

analyzeBtn.addEventListener("click", async () => {
  if (!selectedFile) {
    showError("Please upload an MRI image first.");
    return;
  }

  analyzeBtn.textContent = "Analyzing...";
  analyzeBtn.disabled = true;

  try {
    const formData = new FormData();
    formData.append("image", selectedFile);

    const response = await fetch(`${API_BASE_URL}/predict`, {
      method: "POST",
      body: formData,
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "Prediction failed.");
    }

    predictionText.classList.remove("error-text");
    predictionText.textContent = formatLabel(data.prediction);
    confidenceText.textContent = `${Number(data.percentage || 0).toFixed(2)}%`;

    const probabilities = mapProbabilities(data.probabilities || []);
    updateProbability(gliomaValue, gliomaBar, probabilities.glioma);
    updateProbability(meningiomaValue, meningiomaBar, probabilities.meningioma);
    updateProbability(pituitaryValue, pituitaryBar, probabilities.pituitary);
    updateProbability(noTumorValue, noTumorBar, probabilities.notumor);

    if (data.grad_cam) {
      gradcamImage.src = data.grad_cam;
      gradcamCard.classList.add("has-image");
      gradcamMessage.textContent = "Grad-CAM heatmap generated for the detected tumor class.";
    } else {
      gradcamImage.removeAttribute("src");
      gradcamCard.classList.remove("has-image");
      gradcamMessage.textContent = "No tumor predicted, so Grad-CAM visualization is not generated.";
    }
  } catch (error) {
    showError(error.message);
  } finally {
    analyzeBtn.textContent = "Analyze";
    analyzeBtn.disabled = false;
  }
});

function mapProbabilities(items) {
  const result = {
    glioma: 0,
    meningioma: 0,
    pituitary: 0,
    notumor: 0,
  };

  items.forEach((item) => {
    if (Object.prototype.hasOwnProperty.call(result, item.label)) {
      result[item.label] = Number(item.confidence || 0);
    }
  });

  return result;
}

function updateProbability(textElement, barElement, value) {
  const safeValue = Math.max(0, Math.min(1, Number(value) || 0));
  const percent = safeValue * 100;
  textElement.textContent = `${percent.toFixed(2)}%`;
  barElement.style.width = `${percent}%`;
}

function formatLabel(label) {
  if (!label) {
    return "No result";
  }

  if (label === "notumor") {
    return "No Tumor";
  }

  return label.replace(/\b\w/g, (char) => char.toUpperCase());
}
