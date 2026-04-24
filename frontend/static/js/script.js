const dropArea = document.getElementById("dropArea");
const fileInput = document.getElementById("fileInput");
const fileName = document.getElementById("fileName");
const previewImage = document.getElementById("previewImage");
const uploadContent = document.getElementById("uploadContent");
const analyzeBtn = document.getElementById("analyzeBtn");

const predictionText = document.getElementById("predictionText");
const confidenceText = document.getElementById("confidenceText");
const inferenceTimeText = document.getElementById("inferenceTimeText");
const contextTitle = document.getElementById("contextTitle");
const contextDefinition = document.getElementById("contextDefinition");
const specialistText = document.getElementById("specialistText");
const nextStepText = document.getElementById("nextStepText");
const architectureText = document.getElementById("architectureText");
const modelVersionText = document.getElementById("modelVersionText");
const inputSizeText = document.getElementById("inputSizeText");
const datasetText = document.getElementById("datasetText");
const datasetSourceText = document.getElementById("datasetSourceText");
const trainingNoteText = document.getElementById("trainingNoteText");

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
const CLINICAL_CONTEXT = {
  glioma: {
    title: "Glioma context",
    definition:
      "Glioma refers to a tumor arising from glial cells in the brain or spinal cord. Imaging review typically considers location, enhancement pattern, and surrounding edema before any diagnosis is made.",
    specialist: "Neurosurgeon and Neuro-oncologist",
    nextStep:
      "Compare with formal radiology findings and neurological symptoms, then consider specialist referral for further evaluation.",
  },
  meningioma: {
    title: "Meningioma context",
    definition:
      "Meningioma is usually a tumor arising from the meninges, the layers covering the brain and spinal cord. Imaging interpretation often looks at extra-axial position, dural attachment, and mass effect.",
    specialist: "Neurosurgeon",
    nextStep:
      "Review the scan with a radiologist and neurosurgeon to decide whether interval monitoring, advanced imaging, or further workup is needed.",
  },
  pituitary: {
    title: "Pituitary lesion context",
    definition:
      "Pituitary tumors are growths in or near the pituitary gland and may affect hormones or vision depending on size and location. MRI findings are usually interpreted alongside endocrine symptoms and lab results.",
    specialist: "Endocrinologist and Neurosurgeon",
    nextStep:
      "Correlate the scan with hormone studies, symptoms, and specialist review before drawing conclusions from the image alone.",
  },
  notumor: {
    title: "No tumor estimate context",
    definition:
      "A no-tumor estimate means this model did not find image features strongly matching its tumor classes. It does not rule out disease, subtle findings, or conditions outside the model's training scope.",
    specialist: "Radiologist or Neurologist",
    nextStep:
      "If symptoms persist, rely on formal radiology interpretation and clinical evaluation rather than this screening-style result.",
  },
};

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
  inferenceTimeText.textContent = "--";
  updateClinicalContext(null);
  updateTechnicalDetails(null);

  updateProbability(gliomaValue, gliomaBar, 0);
  updateProbability(meningiomaValue, meningiomaBar, 0);
  updateProbability(pituitaryValue, pituitaryBar, 0);
  updateProbability(noTumorValue, noTumorBar, 0);

  gradcamImage.removeAttribute("src");
  gradcamCard.classList.remove("has-image");
  gradcamMessage.textContent = "Illustrative heatmap appears here after analysis.";
}

function showError(message) {
  predictionText.textContent = message;
  predictionText.classList.add("error-text");
  confidenceText.textContent = "0%";
  inferenceTimeText.textContent = "--";
  updateClinicalContext(null);
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
    inferenceTimeText.textContent = `${Number(data.inference_ms || 0).toFixed(2)} ms`;
    updateClinicalContext(data.prediction);
    updateTechnicalDetails(data.technical_details || null);

    const probabilities = mapProbabilities(data.probabilities || []);
    updateProbability(gliomaValue, gliomaBar, probabilities.glioma);
    updateProbability(meningiomaValue, meningiomaBar, probabilities.meningioma);
    updateProbability(pituitaryValue, pituitaryBar, probabilities.pituitary);
    updateProbability(noTumorValue, noTumorBar, probabilities.notumor);

    if (data.grad_cam) {
      gradcamImage.src = data.grad_cam;
      gradcamCard.classList.add("has-image");
      gradcamMessage.textContent =
        "Grad-CAM heatmap generated as an illustrative explanation for the AI estimate.";
    } else {
      gradcamImage.removeAttribute("src");
      gradcamCard.classList.remove("has-image");
      gradcamMessage.textContent =
        "No tumor estimate was returned, so no Grad-CAM overlay is shown.";
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

function updateClinicalContext(label) {
  const context = CLINICAL_CONTEXT[label] || {
    title: "Awaiting result",
    definition: "Upload an MRI image to view a short explanation for the AI estimate.",
    specialist: "Neurologist / Neurosurgeon",
    nextStep: "Correlate with radiology review and clinical history.",
  };

  contextTitle.textContent = context.title;
  contextDefinition.textContent = context.definition;
  specialistText.textContent = context.specialist;
  nextStepText.textContent = context.nextStep;
}

function updateTechnicalDetails(details) {
  architectureText.textContent = details?.model_name || "EfficientNetB0";
  modelVersionText.textContent = details?.model_version || "best_model.keras";
  inputSizeText.textContent = details?.input_size || "224 x 224 x 3";
  datasetText.textContent = details?.dataset_name || "Brain Tumor MRI Dataset";
  datasetSourceText.textContent = details?.dataset_source || "Project dataset";

  const datasetUsage = details?.dataset_usage_note ? `${details.dataset_usage_note} ` : "";
  const trainingNote = details?.training_metrics_note || "Training metrics are documented in the project notebooks.";
  trainingNoteText.textContent = `${datasetUsage}${trainingNote}`.trim();
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
