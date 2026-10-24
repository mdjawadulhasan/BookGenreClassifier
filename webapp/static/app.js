const form = document.querySelector("#prediction-form");
const featureSelect = document.querySelector("#feature");
const urlInput = document.querySelector("#book-url");
const predictButton = document.querySelector("#predict-button");
const statusText = document.querySelector("#status");
const resultPanel = document.querySelector("#result-panel");
let modelOptions = [];

function showStatus(message, isError = false) {
  statusText.textContent = message;
  statusText.classList.toggle("error", isError);
}

function updateModel() {
  const selected = modelOptions.find(model => model.feature === featureSelect.value);
  document.querySelector("#model-name").textContent = selected.model;
  document.querySelector("#model-accuracy").textContent = `${(selected.accuracy * 100).toFixed(2)}%`;
  document.querySelector("#model-f1").textContent = selected.macro_f1.toFixed(4);
  document.querySelector("#model-card").hidden = false;
}

async function loadModels() {
  try {
    const response = await fetch("/api/models");
    if (!response.ok) throw new Error("Could not load the models. Refresh to try again.");
    const data = await response.json();
    modelOptions = data.models;
    featureSelect.replaceChildren();
    for (const model of modelOptions) {
      const option = document.createElement("option");
      option.value = model.feature;
      option.textContent = model.label;
      featureSelect.append(option);
    }
    featureSelect.value = "title_description";
    featureSelect.disabled = false;
    predictButton.disabled = false;
    updateModel();
    showStatus("Ready when you are.");
  } catch (error) {
    showStatus(error.message, true);
  }
}

function showPrediction(data) {
  // Use textContent for scraped content so website text cannot become HTML.
  document.querySelector("#book-title").textContent = data.book.title;
  document.querySelector("#prediction").textContent = data.prediction;
  const selected = modelOptions.find(model => model.feature === data.feature);
  document.querySelector("#prediction-method").textContent = `${data.model} · ${selected.label} · TF-IDF`;
  document.querySelector("#site-category").textContent = data.book.site_category || "Not available";
  document.querySelector("#source-link").href = data.book.book_url;
  const cover = document.querySelector("#book-cover");
  cover.hidden = true;
  if (data.book.image_url) {
    cover.alt = `Cover of ${data.book.title}`;
    cover.onload = () => { cover.hidden = false; };
    cover.onerror = () => { cover.hidden = true; };
    cover.src = data.book.image_url;
  } else {
    cover.removeAttribute("src");
  }
  document.querySelector("#empty-result").hidden = true;
  document.querySelector("#result").hidden = false;
}

form.addEventListener("submit", async event => {
  event.preventDefault();
  if (!form.reportValidity()) return;
  predictButton.disabled = true;
  featureSelect.disabled = true;
  resultPanel.setAttribute("aria-busy", "true");
  // Clear an earlier result while the new request is being processed.
  document.querySelector("#result").hidden = true;
  document.querySelector("#empty-result").hidden = false;
  showStatus("Fetching the book page and making a prediction...");
  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({book_url: urlInput.value.trim(), feature: featureSelect.value})
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(typeof data.detail === "string" ? data.detail : "Please check the book link and feature option.");
    }
    showPrediction(data);
    showStatus("Prediction complete.");
  } catch (error) {
    showStatus(error.message || "Something went wrong. Please try again.", true);
  } finally {
    predictButton.disabled = false;
    featureSelect.disabled = false;
    resultPanel.setAttribute("aria-busy", "false");
  }
});

featureSelect.addEventListener("change", updateModel);
document.querySelector("#example-fiction").addEventListener("click", () => {
  urlInput.value = "https://books.toscrape.com/catalogue/thirst_946/index.html";
  urlInput.focus();
});
document.querySelector("#example-nonfiction").addEventListener("click", () => {
  urlInput.value = "https://books.toscrape.com/catalogue/the-five-love-languages-how-to-express-heartfelt-commitment-to-your-mate_969/index.html";
  urlInput.focus();
});

loadModels();
