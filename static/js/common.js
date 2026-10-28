"use strict";

const DHT = {
  async api(url, options = {}) {
    const response = await fetch(url, {
      headers: {"Content-Type": "application/json", ...(options.headers || {})},
      ...options
    });
    let body;
    try { body = await response.json(); } catch (_) { body = {error: `HTTP ${response.status}`}; }
    if (!response.ok || body.ok === false) throw new Error(body.error || `Request failed (${response.status})`);
    return body;
  },
  toast(message, isError = false) {
    const node = document.getElementById("toast");
    node.textContent = message;
    node.className = `toast show${isError ? " error" : ""}`;
    clearTimeout(DHT.toastTimer);
    DHT.toastTimer = setTimeout(() => node.className = "toast", 4000);
  },
  number(value, digits = 2) {
    return Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : "—";
  },
  drawChart(canvas, series) {
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const width = canvas.width, height = canvas.height;
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#fbfbf7"; ctx.fillRect(0, 0, width, height);
    const all = series.flatMap(item => item.values).filter(Number.isFinite);
    if (!all.length) {
      ctx.fillStyle = "#879089"; ctx.font = "14px system-ui"; ctx.textAlign = "center";
      ctx.fillText("No data to plot", width / 2, height / 2); return;
    }
    const pad = {left: 55, right: 25, top: 28, bottom: 35};
    const min = Math.min(...all), max = Math.max(...all), range = max - min || 1;
    ctx.strokeStyle = "#e1e4de"; ctx.lineWidth = 1; ctx.fillStyle = "#7c867f"; ctx.font = "11px monospace"; ctx.textAlign = "right";
    for (let i = 0; i <= 4; i++) {
      const y = pad.top + (height - pad.top - pad.bottom) * i / 4;
      ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(width - pad.right, y); ctx.stroke();
      ctx.fillText((max - range * i / 4).toFixed(1), pad.left - 9, y + 4);
    }
    const longest = Math.max(...series.map(item => item.values.length));
    series.forEach(item => {
      ctx.strokeStyle = item.color; ctx.lineWidth = 3; ctx.lineJoin = "round"; ctx.beginPath();
      item.values.forEach((value, index) => {
        const x = pad.left + (width - pad.left - pad.right) * (longest <= 1 ? .5 : index / (longest - 1));
        const y = pad.top + (height - pad.top - pad.bottom) * (1 - (value - min) / range);
        index ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
      }); ctx.stroke();
    });
    ctx.textAlign = "left";
    series.forEach((item, index) => {ctx.fillStyle = item.color; ctx.fillRect(pad.left + index * 150, 9, 18, 3); ctx.fillStyle = "#59645d"; ctx.fillText(item.label, pad.left + 24 + index * 150, 14);});
  }
};

(async () => {
  const status = document.getElementById("serverStatus");
  try {
    const health = await DHT.api("/api/health");
    const stores = [health.mysql.ready ? "MySQL" : null, health.firebase.ready ? "Firebase" : null].filter(Boolean);
    status.className = "status online";
    status.innerHTML = `<i></i> Flask online${stores.length ? ` · ${stores.join(" + ")}` : " · demo storage"}`;
  } catch (error) {
    status.className = "status offline"; status.innerHTML = "<i></i> Server unavailable";
  }
})();
