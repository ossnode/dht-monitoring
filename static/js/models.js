"use strict";

const page = document.body.dataset.page;
let trainedModelId = null;

function field(id, fallback) { const node = document.getElementById(id); return node ? node.value : fallback; }
function renderMetrics(metrics) {
  const labels = {mae: "MAE", rmse: "RMSE", r2: "R²", mape_percent: "MAPE"};
  document.getElementById("metrics").innerHTML = Object.entries(labels).map(([key, label]) => `<div class="metric"><small>${label}</small><strong>${metrics[key] == null ? "n/a" : DHT.number(metrics[key], 4)}${key === "mape_percent" && metrics[key] != null ? "%" : ""}</strong></div>`).join("");
}
function renderFuture(values) {
  const node = document.getElementById("futurePredictions");
  node.className = "prediction-list";
  node.innerHTML = values.map((value, index) => `<span>t+${index + 1}: ${DHT.number(value, 3)}</span>`).join("");
}
function detailsFor(result) {
  if (page === "mlr") {
    const coefficients = Object.entries(result.coefficients).map(([name, value]) => `<span>${name}: ${DHT.number(value, 5)}</span>`).join("");
    return `<div class="equation">${result.equation}</div><div class="coefficient-grid">${coefficients}</div>`;
  }
  if (page === "mlp") return `<div class="equation">Architecture: ${result.hidden_layers.join(" → ")} · epochs completed: ${result.epochs_completed} · final loss: ${DHT.number(result.loss, 6)}</div>`;
  const importance = Object.entries(result.feature_importance || {}).sort((a,b) => b[1] - a[1]).map(([name, value]) => `<span>${name}: ${DHT.number(value, 4)}</span>`).join("");
  return `<p class="eyebrow">FEATURE IMPORTANCE</p><div class="coefficient-grid">${importance}</div>`;
}
async function train() {
  const button = document.getElementById("trainButton");
  button.disabled = true; button.textContent = page === "mlr" ? "Analysing…" : "Training…";
  document.getElementById("modelState").textContent = "Working";
  const common = {target: field("target"), lags: Number(field("lags", 3))};
  let endpoint, payload;
  if (page === "mlr") {
    endpoint = "/api/mlr/analyse"; payload = {...common, steps: Number(field("steps", 5))};
  } else if (page === "mlp") {
    endpoint = "/api/mlp/train"; payload = {...common, hidden_layers: field("hiddenLayers", "32,16").split(",").map(Number), activation: field("activation", "relu"), epochs: Number(field("epochs", 500)), learning_rate: Number(field("learningRate", .001))};
  } else {
    endpoint = "/api/third-model/train"; payload = {...common, decision_width: Number(field("decisionWidth", 8)), attention_steps: Number(field("attentionSteps", 3)), epochs: Number(field("epochs", 100))};
  }
  try {
    const result = await DHT.api(endpoint, {method: "POST", body: JSON.stringify(payload)});
    trainedModelId = result.model_id;
    renderMetrics(result.metrics);
    document.getElementById("modelDetails").innerHTML = detailsFor(result);
    document.getElementById("modelState").textContent = "Ready";
    DHT.drawChart(document.getElementById("predictionChart"), [
      {label: "Actual", color: "#184c3a", values: result.validation.actual.map(Number)},
      {label: "Predicted", color: page === "mlp" ? "#8b75ec" : page === "third" ? "#3ba8a3" : "#e6982d", values: result.validation.predicted.map(Number)}
    ]);
    if (result.future_predictions) renderFuture(result.future_predictions);
    DHT.toast(`${result.model} completed successfully`);
  } catch (error) {
    document.getElementById("modelState").textContent = "Error"; DHT.toast(error.message, true);
  } finally {
    button.disabled = false; button.textContent = page === "mlr" ? "Analyse with MLR" : page === "mlp" ? "Train MLP" : "Train TabNet";
  }
}

async function predict() {
  if (!trainedModelId) return DHT.toast("Train the model first", true);
  const endpoint = page === "mlp" ? "/api/mlp/predict" : "/api/third-model/predict";
  try {
    const result = await DHT.api(endpoint, {method: "POST", body: JSON.stringify({model_id: trainedModelId, steps: Number(field("predictSteps", 5))})});
    renderFuture(result.predictions);
  } catch (error) { DHT.toast(error.message, true); }
}

document.getElementById("trainButton").addEventListener("click", train);
const predictButton = document.getElementById("predictButton");
if (predictButton) predictButton.addEventListener("click", predict);
DHT.drawChart(document.getElementById("predictionChart"), []);
