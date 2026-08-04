/* NIDS web app — theme toggle, manual-flow form, upload dropzone. */
(function () {
  "use strict";

  /* ---------------------------------------------------------------- theme */

  var root = document.documentElement;
  var toggle = document.querySelector("[data-theme-toggle]");

  function currentTheme() {
    var stamped = root.getAttribute("data-theme");
    if (stamped) return stamped;
    return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }

  if (toggle) {
    toggle.addEventListener("click", function () {
      var next = currentTheme() === "dark" ? "light" : "dark";
      root.setAttribute("data-theme", next);
      try { localStorage.setItem("ids-theme", next); } catch (e) { /* private mode */ }
    });
  }

  /* ------------------------------------------------------------- helpers */

  function percent(x) { return (x * 100).toFixed(2) + "%"; }

  function el(tag, className, text) {
    var node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function cookie(name) {
    var match = document.cookie.match(new RegExp("(^| )" + name + "=([^;]+)"));
    return match ? decodeURIComponent(match[2]) : "";
  }

  /* -------------------------------------------------- manual flow form */

  var form = document.getElementById("flow-form");
  if (!form) return;

  var presetsNode = document.getElementById("presets-data");
  var presets = presetsNode ? JSON.parse(presetsNode.textContent) : {};
  var verdict = document.getElementById("verdict");
  var submitBtn = document.getElementById("submit-btn");

  function inputs() {
    return Array.prototype.slice.call(form.querySelectorAll('input[type="number"]'));
  }

  function setValues(getter) {
    inputs().forEach(function (input) {
      var v = getter(input);
      input.value = v === null || v === undefined ? "" : v;
      input.closest(".field").classList.remove("has-error");
    });
  }

  var presetSelect = document.getElementById("preset");
  if (presetSelect) {
    presetSelect.addEventListener("change", function () {
      var row = presets[presetSelect.value];
      if (!row) return;
      setValues(function (input) {
        var v = row[input.name];
        return v === undefined ? "" : Math.round(v * 1e6) / 1e6;
      });
    });
  }

  var fillMedian = document.getElementById("fill-median");
  if (fillMedian) {
    fillMedian.addEventListener("click", function () {
      setValues(function (input) { return input.getAttribute("placeholder"); });
      if (presetSelect) presetSelect.value = "";
    });
  }

  var clearAll = document.getElementById("clear-all");
  if (clearAll) {
    clearAll.addEventListener("click", function () {
      setValues(function () { return ""; });
      if (presetSelect) presetSelect.value = "";
      renderEmpty();
    });
  }

  /* --------------------------------------------------------- rendering */

  function renderEmpty() {
    verdict.replaceChildren(
      el("p", "verdict-empty", "Fill in the features and run the classifier to see a prediction.")
    );
  }

  function renderError(message, fields) {
    verdict.replaceChildren();

    var box = el("div", "notice is-error");
    box.appendChild(el("span", null, message));
    verdict.appendChild(box);

    inputs().forEach(function (input) {
      input.closest(".field").classList.toggle("has-error", Boolean(fields && fields[input.name]));
    });

    if (fields && fields[Object.keys(fields)[0]]) {
      var first = form.querySelector('.field.has-error input');
      if (first) first.focus({ preventScroll: false });
    }
  }

  function renderResult(data) {
    verdict.replaceChildren();

    // Verdict banner — colour is always paired with an icon and the class name,
    // never carrying the meaning on its own.
    var banner = el("div", "verdict-banner " + (data.is_attack ? "is-attack" : "is-benign"));
    banner.appendChild(el("span", "dot"));

    var textWrap = el("div");
    textWrap.appendChild(el("div", "label", data.label));
    textWrap.appendChild(el("div", "sub", data.is_attack ? "Attack traffic" : "Normal traffic"));
    banner.appendChild(textWrap);
    verdict.appendChild(banner);

    var conf = el("div");
    conf.appendChild(el("div", "field-label", "Confidence"));
    conf.appendChild(el("div", "confidence-value", percent(data.confidence)));
    conf.appendChild(el("div", "field-hint", "predicted by " + data.model));
    verdict.appendChild(conf);

    // Top-3 probabilities as a single-series bar chart with direct labels.
    var bars = el("div", "bars");
    data.top.forEach(function (row, i) {
      var isBenign = row.label === "Benign";
      var barRow = el("div", "bar-row " + (i === 0 ? (isBenign ? "is-benign" : "is-attack") : ""));

      var meta = el("div", "bar-meta");
      meta.appendChild(el("span", "name", row.label));
      meta.appendChild(el("span", "value", percent(row.confidence)));
      barRow.appendChild(meta);

      var track = el("div", "bar-track");
      var fill = el("div", "bar-fill");
      fill.style.width = "0%";
      track.appendChild(fill);
      barRow.appendChild(track);
      bars.appendChild(barRow);

      requestAnimationFrame(function () {
        fill.style.width = Math.max(row.confidence * 100, 0.6) + "%";
      });
    });

    var barsWrap = el("div");
    barsWrap.appendChild(el("div", "field-label", "Most likely classes"));
    barsWrap.appendChild(bars);
    verdict.appendChild(barsWrap);
  }

  /* ------------------------------------------------------------ submit */

  function busy(state) {
    submitBtn.disabled = state;
    submitBtn.replaceChildren();
    if (state) {
      submitBtn.appendChild(el("span", "spinner"));
      submitBtn.appendChild(el("span", null, "Classifying…"));
    } else {
      submitBtn.appendChild(el("span", null, "Classify flow"));
    }
  }

  form.addEventListener("submit", function (event) {
    event.preventDefault();

    var missing = inputs().filter(function (i) { return i.value.trim() === ""; });
    if (missing.length) {
      var fieldErrors = {};
      missing.forEach(function (i) { fieldErrors[i.name] = true; });
      renderError(
        missing.length + " of 30 features are empty. Use “Fill with medians” or load an example flow.",
        fieldErrors
      );
      return;
    }

    var payload = {};
    inputs().forEach(function (i) { payload[i.name] = i.value; });
    payload.model_key = form.querySelector("#model_key").value;

    busy(true);

    fetch("/api/predict/", {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-CSRFToken": cookie("csrftoken") },
      body: JSON.stringify(payload)
    })
      .then(function (r) { return r.json().then(function (b) { return { ok: r.ok, body: b }; }); })
      .then(function (res) {
        if (!res.ok) {
          renderError(res.body.error || "Prediction failed.", res.body.fields);
          return;
        }
        renderResult(res.body);
      })
      .catch(function () {
        renderError("Could not reach the server. Is it still running?");
      })
      .finally(function () { busy(false); });
  });

  /* ------------------------------------------------------- dropzone */

  var dropzone = document.querySelector("[data-dropzone]");
  if (dropzone) {
    var fileInput = dropzone.querySelector('input[type="file"]');
    var nameLabel = dropzone.querySelector("[data-filename]");

    function showFile(file) {
      if (!file) return;
      dropzone.classList.add("has-file");
      if (nameLabel) {
        nameLabel.textContent = file.name + " · " + (file.size / 1048576).toFixed(2) + " MB";
      }
    }

    dropzone.addEventListener("click", function () { fileInput.click(); });
    fileInput.addEventListener("change", function () { showFile(fileInput.files[0]); });

    ["dragenter", "dragover"].forEach(function (evt) {
      dropzone.addEventListener(evt, function (e) {
        e.preventDefault();
        dropzone.classList.add("is-over");
      });
    });
    ["dragleave", "drop"].forEach(function (evt) {
      dropzone.addEventListener(evt, function (e) {
        e.preventDefault();
        dropzone.classList.remove("is-over");
      });
    });
    dropzone.addEventListener("drop", function (e) {
      if (e.dataTransfer.files.length) {
        fileInput.files = e.dataTransfer.files;
        showFile(fileInput.files[0]);
      }
    });
  }
})();
