const state = {
  file: null,
  prediction: null,
  feedbackChoice: null,
};

const elements = {
  dropzone: document.querySelector("#dropzone"),
  fileInput: document.querySelector("#fileInput"),
  chooseButton: document.querySelector("#chooseButton"),
  preview: document.querySelector("#uploadPreview"),
  previewImage: document.querySelector("#previewImage"),
  previewName: document.querySelector("#previewName"),
  previewMeta: document.querySelector("#previewMeta"),
  predictButton: document.querySelector("#predictButton"),
  uploadError: document.querySelector("#uploadError"),
  resultSection: document.querySelector("#resultSection"),
  resultSource: document.querySelector("#resultSource"),
  top1Label: document.querySelector("#top1Label"),
  top1Probability: document.querySelector("#top1Probability"),
  top1Track: document.querySelector("#top1Track"),
  predictionDisclaimer: document.querySelector("#predictionDisclaimer"),
  candidateList: document.querySelector("#candidateList"),
  knowledgeCount: document.querySelector("#knowledgeCount"),
  knowledgeSourceNote: document.querySelector("#knowledgeSourceNote"),
  knowledgeRecords: document.querySelector("#knowledgeRecords"),
  relatedImages: document.querySelector("#relatedImages"),
  correctButton: document.querySelector("#correctButton"),
  incorrectButton: document.querySelector("#incorrectButton"),
  correctionPanel: document.querySelector("#correctionPanel"),
  correctLabel: document.querySelector("#correctLabel"),
  submitCorrection: document.querySelector("#submitCorrection"),
  feedbackMessage: document.querySelector("#feedbackMessage"),
  globalError: document.querySelector("#globalError"),
};

const allowedExtensions = new Set([".jpg", ".jpeg", ".png"]);

function setMessage(element, message, isError = false) {
  element.textContent = message;
  element.classList.toggle("message-error", isError);
  element.classList.toggle("is-hidden", !message);
}

function resetFeedback() {
  state.feedbackChoice = null;
  elements.correctButton.classList.remove("is-selected");
  elements.incorrectButton.classList.remove("is-selected");
  elements.correctionPanel.classList.add("is-hidden");
  elements.feedbackMessage.classList.add("is-hidden");
}

function selectFile(file) {
  if (!file) return;
  const extension = `.${file.name.split(".").pop().toLowerCase()}`;
  if (!allowedExtensions.has(extension)) {
    state.file = null;
    elements.preview.classList.add("is-hidden");
    setMessage(
      elements.uploadError,
      "暂不支持该图片格式，请上传 JPG、JPEG 或 PNG 图片。",
      true
    );
    return;
  }
  if (file.size > 10 * 1024 * 1024) {
    state.file = null;
    elements.preview.classList.add("is-hidden");
    setMessage(elements.uploadError, "图片文件过大，请上传 10 MB 以内的图片。", true);
    return;
  }

  state.file = file;
  setMessage(elements.uploadError, "");
  const objectUrl = URL.createObjectURL(file);
  elements.previewImage.src = objectUrl;
  elements.previewName.textContent = file.name;
  elements.previewMeta.textContent = `${Math.round(file.size / 1024)} KB · ${extension.slice(1).toUpperCase()}`;
  elements.preview.classList.remove("is-hidden");
  elements.predictButton.disabled = false;
}

function formatProbability(value) {
  return `${(Number(value) * 100).toFixed(1)}%`;
}

function renderPrediction(result) {
  state.prediction = result;
  resetFeedback();
  elements.resultSection.classList.remove("is-hidden");
  elements.resultSource.textContent = `${result.model} · ${result.dataset}`;
  elements.top1Label.textContent = result.top1.label;
  elements.top1Probability.textContent = formatProbability(result.top1.probability);
  elements.top1Track.style.width = `${Math.min(100, result.top1.probability * 100)}%`;
  elements.predictionDisclaimer.textContent = result.disclaimer;

  elements.candidateList.innerHTML = result.top3
    .map(
      (candidate) => `
        <div class="candidate-row">
          <span class="candidate-rank">0${candidate.rank}</span>
          <span class="candidate-name">${escapeHtml(candidate.label)}</span>
          <span class="candidate-bar" aria-hidden="true">
            <span style="width:${Math.min(100, candidate.probability * 100)}%"></span>
          </span>
          <span class="candidate-probability">${formatProbability(candidate.probability)}</span>
        </div>
      `
    )
    .join("");

  const knowledge = result.knowledge;
  elements.knowledgeCount.textContent = `${knowledge.record_count} 条资料`;
  elements.knowledgeSourceNote.textContent = knowledge.source_note;
  elements.knowledgeRecords.innerHTML = knowledge.records
    .map(
      (record) => `
        <article class="knowledge-record">
          <h3>${escapeHtml(record.role || "未填写角色")}</h3>
          <span class="opera">${escapeHtml(record.opera || "剧目未填写")} · 资料编号 ${escapeHtml(record.metadata_id)}</span>
          <div class="knowledge-fields">
            ${record.fields
              .slice(0, 7)
              .map(
                (field) => `
                  <div class="knowledge-field">
                    <strong>${escapeHtml(field.field)}</strong>
                    <span>${escapeHtml(field.value)}</span>
                  </div>
                `
              )
              .join("")}
          </div>
        </article>
      `
    )
    .join("");

  elements.relatedImages.innerHTML = result.related_images
    .map(
      (image) => `
        <article class="related-card">
          <img src="${image.image_url}" alt="${escapeHtml(image.label)} ${escapeHtml(image.role)}" loading="lazy" />
          <div class="related-card-copy">
            <strong>${escapeHtml(image.role || image.label)}</strong>
            <span>${escapeHtml(image.opera || image.label)}</span>
          </div>
        </article>
      `
    )
    .join("");

  fillCorrectionOptions(result.top3.map((candidate) => candidate.label));
  elements.resultSection.scrollIntoView({ behavior: "smooth", block: "start" });
}

function fillCorrectionOptions(preferredLabels) {
  const labels = [
    "三块瓦脸",
    "碎脸",
    "象形脸",
    "花三块瓦脸",
    "整脸",
    "十字门脸",
    "六分脸",
  ];
  elements.correctLabel.innerHTML = labels
    .map(
      (label) =>
        `<option value="${label}" ${preferredLabels.includes(label) ? "selected" : ""}>${label}</option>`
    )
    .join("");
}

async function runPrediction() {
  if (!state.file) {
    setMessage(elements.uploadError, "请先上传一张图片。", true);
    return;
  }
  elements.predictButton.disabled = true;
  elements.predictButton.innerHTML = "<span aria-hidden=\"true\">…</span> 识别中";
  setMessage(elements.uploadError, "");
  setMessage(elements.globalError, "");
  const formData = new FormData();
  formData.append("file", state.file);
  try {
    const response = await fetch("/api/predict", { method: "POST", body: formData });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "当前图片暂时无法完成识别，请稍后重试。");
    renderPrediction(payload);
  } catch (error) {
    setMessage(elements.globalError, error.message, true);
  } finally {
    elements.predictButton.disabled = false;
    elements.predictButton.innerHTML = "<span aria-hidden=\"true\">↗</span> 开始识别";
  }
}

function chooseFeedback(isCorrect) {
  if (!state.prediction) return;
  state.feedbackChoice = isCorrect;
  elements.correctButton.classList.toggle("is-selected", isCorrect);
  elements.incorrectButton.classList.toggle("is-selected", !isCorrect);
  elements.correctionPanel.classList.toggle("is-hidden", isCorrect);
  if (isCorrect) submitFeedback();
}

async function submitFeedback() {
  if (!state.prediction || state.feedbackChoice === null) return;
  if (!state.feedbackChoice && !elements.correctLabel.value) return;
  elements.correctButton.disabled = true;
  elements.incorrectButton.disabled = true;
  elements.submitCorrection.disabled = true;
  const body = {
    upload_id: state.prediction.upload_id,
    top1: state.prediction.top1,
    top3: state.prediction.top3,
    top1_correct: state.feedbackChoice,
    corrected_label: state.feedbackChoice ? null : elements.correctLabel.value,
  };
  try {
    const response = await fetch("/api/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "反馈暂时无法保存，请稍后重试。");
    setMessage(elements.feedbackMessage, "感谢反馈，已保存到本地反馈记录。");
  } catch (error) {
    setMessage(elements.feedbackMessage, error.message, true);
  } finally {
    elements.correctButton.disabled = false;
    elements.incorrectButton.disabled = false;
    elements.submitCorrection.disabled = false;
  }
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

elements.chooseButton.addEventListener("click", () => elements.fileInput.click());
elements.dropzone.addEventListener("click", (event) => {
  if (event.target.closest("button")) return;
  elements.fileInput.click();
});
elements.dropzone.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    elements.fileInput.click();
  }
});
elements.fileInput.addEventListener("change", (event) => selectFile(event.target.files[0]));
elements.predictButton.addEventListener("click", runPrediction);
elements.correctButton.addEventListener("click", () => chooseFeedback(true));
elements.incorrectButton.addEventListener("click", () => chooseFeedback(false));
elements.submitCorrection.addEventListener("click", submitFeedback);

["dragenter", "dragover"].forEach((eventName) => {
  elements.dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    elements.dropzone.classList.add("is-dragover");
  });
});
["dragleave", "drop"].forEach((eventName) => {
  elements.dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    elements.dropzone.classList.remove("is-dragover");
  });
});
elements.dropzone.addEventListener("drop", (event) => selectFile(event.dataTransfer.files[0]));

