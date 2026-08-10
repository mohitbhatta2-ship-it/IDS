/*
 * Live-capture console controller.
 *
 * Start / Stop drive the /live/start/ and /live/stop/ endpoints; while running,
 * the page polls /live/status/?since=<seq> and appends any new classified flows
 * to the table. All network state lives on the server (the CaptureSession); this
 * file only reflects it. Loaded only on the Live page, so it makes no
 * assumptions about other pages' markup.
 */
(function () {
  "use strict";

  var startBtn = document.getElementById("live-start");
  var stopBtn = document.getElementById("live-stop");
  if (!startBtn || !stopBtn) return; // not the live page

  var modelSel = document.getElementById("live-model");
  var ifaceSel = document.getElementById("live-iface");
  var stateChip = document.getElementById("live-state");
  var stateText = stateChip.querySelector("[data-state-text]");
  var errorBox = document.getElementById("live-error");
  var errorText = errorBox.querySelector("[data-error-text]");
  var tbody = document.getElementById("live-tbody");
  var emptyRow = document.getElementById("live-empty");

  var POLL_MS = 1500;
  var timer = null;
  var lastSeq = 0;

  function cookie(name) {
    var m = document.cookie.match(new RegExp("(^| )" + name + "=([^;]+)"));
    return m ? decodeURIComponent(m[2]) : "";
  }

  function metric(name) { return document.querySelector('[data-metric="' + name + '"]'); }

  function setMetric(name, value) {
    var node = metric(name);
    if (node) node.textContent = value;
  }

  function showError(message) {
    if (!message) { errorBox.hidden = true; return; }
    errorText.textContent = message;
    errorBox.hidden = false;
  }

  function setRunning(running) {
    startBtn.disabled = running || startBtn.hasAttribute("data-unsupported");
    stopBtn.disabled = !running;
    modelSel.disabled = running;
    ifaceSel.disabled = running;
    stateChip.classList.toggle("is-recording", running);
    stateText.textContent = running ? "Recording" : "Idle";
  }

  function pad(n) { return n < 10 ? "0" + n : "" + n; }

  function clockFromISO(iso) {
    // Show HH:MM:SS in local time; fall back to the raw string if unparseable.
    var d = new Date(iso);
    if (isNaN(d.getTime())) return iso || "";
    return pad(d.getHours()) + ":" + pad(d.getMinutes()) + ":" + pad(d.getSeconds());
  }

  function esc(s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }

  function rowFor(rec) {
    var tr = document.createElement("tr");
    tr.className = "fam-" + rec.family + (rec.is_attack ? " is-attack-row" : "");

    var pct = (rec.confidence * 100).toFixed(1) + "%";
    var verdict =
      '<span class="live-dot" style="background: var(--fam);"></span>' +
      '<span class="live-verdict">' + esc(rec.label) + "</span>";

    tr.innerHTML =
      "<td>" + esc(clockFromISO(rec.at)) + "</td>" +
      "<td><code>" + esc(rec.src_ip) + ":" + esc(rec.src_port) + "</code></td>" +
      "<td><code>" + esc(rec.dst_ip) + ":" + esc(rec.dst_port) + "</code></td>" +
      "<td>" + esc(rec.protocol) + "</td>" +
      '<td class="num">' + esc(rec.packets) + "</td>" +
      "<td>" + verdict + "</td>" +
      '<td class="num">' + pct + "</td>";
    return tr;
  }

  function appendRecords(records) {
    if (!records || !records.length) return;
    if (emptyRow && emptyRow.parentNode) emptyRow.remove();

    // Records arrive newest-first; insert so the newest ends up on top.
    records.forEach(function (rec) {
      if (rec.seq > lastSeq) lastSeq = rec.seq;
      tbody.insertBefore(rowFor(rec), tbody.firstChild);
    });

    // Keep the DOM bounded to match the server's buffer.
    while (tbody.children.length > 200) tbody.removeChild(tbody.lastChild);
  }

  function applyStatus(data) {
    setMetric("packets", data.packets || 0);
    setMetric("flows", data.flows || 0);
    setMetric("attacks", data.attacks || 0);
    setMetric("open", data.open_flows || 0);
    appendRecords(data.recent);
    showError(data.error || null);
    setRunning(Boolean(data.running));
    if (!data.running) stopPolling();
  }

  function poll() {
    fetch("/live/status/?since=" + lastSeq, { headers: { "Accept": "application/json" } })
      .then(function (r) { return r.json(); })
      .then(applyStatus)
      .catch(function () { /* transient; the next tick retries */ });
  }

  function startPolling() {
    if (timer) return;
    timer = setInterval(poll, POLL_MS);
  }

  function stopPolling() {
    if (timer) { clearInterval(timer); timer = null; }
  }

  startBtn.addEventListener("click", function () {
    showError(null);
    startBtn.disabled = true;
    var payload = { iface: ifaceSel.value, model_key: modelSel.value };

    fetch("/live/start/", {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-CSRFToken": cookie("csrftoken") },
      body: JSON.stringify(payload)
    })
      .then(function (r) { return r.json().then(function (d) { return { ok: r.ok, data: d }; }); })
      .then(function (res) {
        if (!res.ok) {
          showError(res.data.error || "Could not start capture.");
          setRunning(false);
          return;
        }
        lastSeq = 0;
        applyStatus(res.data);
        startPolling();
      })
      .catch(function () {
        showError("Could not reach the server to start capture.");
        setRunning(false);
      });
  });

  stopBtn.addEventListener("click", function () {
    stopBtn.disabled = true;
    fetch("/live/stop/", {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-CSRFToken": cookie("csrftoken") }
    })
      .then(function (r) { return r.json(); })
      .then(function (data) { applyStatus(data); stopPolling(); })
      .catch(function () { showError("Could not reach the server to stop capture."); });
  });

  // If the server already has a running session (e.g. after a page refresh),
  // reflect it and resume polling.
  if (startBtn.hasAttribute("disabled") && startBtn.getAttribute("disabled") !== null) {
    // Disabled at render only when capture is unsupported; mark it so setRunning
    // never re-enables it.
    startBtn.setAttribute("data-unsupported", "1");
  }
  poll();
  fetch("/live/status/", { headers: { "Accept": "application/json" } })
    .then(function (r) { return r.json(); })
    .then(function (data) { if (data.running) startPolling(); })
    .catch(function () {});
})();
