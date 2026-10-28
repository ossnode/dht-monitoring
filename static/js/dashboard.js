"use strict";

async function loadDashboard() {
  try {
    const data = await DHT.api("/api/readings?limit=100");
    const rows = data.readings;
    document.getElementById("readingCount").textContent = rows.length;
    document.getElementById("dataSource").textContent = data.source;
    if (rows.length) {
      const latest = rows[rows.length - 1];
      document.getElementById("latestTemp").textContent = `${DHT.number(latest.temperature, 1)} °C`;
      document.getElementById("latestHumidity").textContent = `${DHT.number(latest.humidity, 1)} %`;
      document.getElementById("orbTemp").textContent = DHT.number(latest.temperature, 1);
      document.getElementById("lastUpdated").textContent = new Date(latest.timestamp_ms).toLocaleString();
    }
    document.getElementById("readingsBody").innerHTML = rows.length ? [...rows].reverse().slice(0, 25).map(row => `<tr><td>${new Date(row.timestamp_ms).toLocaleString()}</td><td>${DHT.number(row.temperature)} °C</td><td>${DHT.number(row.humidity)} %</td><td><code>${row.id || "buffer"}</code></td></tr>`).join("") : '<tr><td colspan="4" class="empty">No readings yet. Send a reading to <code>POST /data</code>.</td></tr>';
    DHT.drawChart(document.getElementById("readingsChart"), [
      {label: "Temperature °C", color: "#e6982d", values: rows.map(row => Number(row.temperature))},
      {label: "Humidity %", color: "#4faeaa", values: rows.map(row => Number(row.humidity))}
    ]);
  } catch (error) { DHT.toast(error.message, true); }
}

document.getElementById("refreshReadings").addEventListener("click", loadDashboard);
document.getElementById("preprocessButton").addEventListener("click", async () => {
  const operations = [...document.querySelectorAll('input[name="operation"]:checked')].map(input => input.value);
  const button = document.getElementById("preprocessButton");
  button.disabled = true;
  try {
    const result = await DHT.api("/api/preprocess", {method: "POST", body: JSON.stringify({operations, window: 3})});
    document.getElementById("preprocessSummary").textContent = `${result.summary.output} rows`;
    const notes = result.summary.notes.length ? ` ${result.summary.notes.join(" · ")}` : "";
    document.getElementById("preprocessOutput").textContent = `Python received ${result.summary.input} rows and returned ${result.summary.output}; ${result.summary.removed} removed.${notes}`;
    DHT.toast("Preprocessing completed on the Flask server");
  } catch (error) { DHT.toast(error.message, true); }
  finally { button.disabled = false; }
});
loadDashboard();
setInterval(loadDashboard, 5000);
