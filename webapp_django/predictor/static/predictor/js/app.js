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

  // requestAnimationFrame never fires while a tab is in the background, so a
  // page opened in a background tab would keep its bars at 0% and its ring
  // empty. Fall back to a timer when the document is hidden.
  function nextFrame(fn) {
    if (document.visibilityState === "hidden") { setTimeout(fn, 0); return; }
    requestAnimationFrame(fn);
  }

  function countUp(node, target, duration) {
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches ||
        document.visibilityState === "hidden") {
      node.textContent = target.toFixed(1);
      return;
    }
    const start = performance.now();
    (function step(now) {
      const t = Math.min((now - start) / duration, 1);
      const eased = 1 - Math.pow(1 - t, 3);
      node.textContent = (target * eased).toFixed(1);
      if (t < 1) requestAnimationFrame(step);
    })(start);
  }

  function markFilled(input) {
    input.closest(".field").classList.toggle("filled", input.value.trim() !== "");
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
      markFilled(input);
    });
  }

  inputs().forEach(function (i) {
    markFilled(i);
    i.addEventListener("input", function () { markFilled(i); });
  });

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

  function ring(fraction, family) {
    // Confidence as a stroke-dashoffset arc. Built with createElementNS because
    // innerHTML does not work for SVG children.
    const NS = "http://www.w3.org/2000/svg";
    const R = 44, C = 2 * Math.PI * R;

    const wrap = el("div", "ring " + family);
    const svg = document.createElementNS(NS, "svg");
    svg.setAttribute("viewBox", "0 0 104 104");
    svg.setAttribute("width", "116");
    svg.setAttribute("height", "116");

    // Gradient along the arc, from a translucent form of the family colour to
    // the solid one, so the ring reads as filling rather than as a flat band.
    const defs = document.createElementNS(NS, "defs");
    const grad = document.createElementNS(NS, "linearGradient");
    grad.setAttribute("id", "ringGrad");
    grad.setAttribute("x1", "0"); grad.setAttribute("y1", "0");
    grad.setAttribute("x2", "1"); grad.setAttribute("y2", "1");
    [["0%", ".45"], ["55%", ".85"], ["100%", "1"]].forEach(function (p) {
      const stop = document.createElementNS(NS, "stop");
      stop.setAttribute("offset", p[0]);
      stop.setAttribute("stop-color", "currentColor");
      stop.setAttribute("stop-opacity", p[1]);
      grad.appendChild(stop);
    });
    defs.appendChild(grad);
    svg.appendChild(defs);

    ["track", "value"].forEach(function (role) {
      const c = document.createElementNS(NS, "circle");
      c.setAttribute("cx", "52"); c.setAttribute("cy", "52"); c.setAttribute("r", String(R));
      c.setAttribute("fill", "none"); c.setAttribute("stroke-width", "9");
      c.setAttribute("class", role);
      if (role === "value") {
        c.setAttribute("stroke-dasharray", String(C));
        c.setAttribute("stroke-dashoffset", String(C));
        nextFrame(function () {
          c.setAttribute("stroke-dashoffset", String(C * (1 - Math.max(fraction, 0.005))));
        });
      }
      svg.appendChild(c);
    });

    wrap.appendChild(svg);

    const label = el("div", "ring-label");
    const n = el("div", "n", "0.0");
    label.appendChild(n);
    label.appendChild(el("div", "u", "percent"));
    wrap.appendChild(label);

    countUp(n, fraction * 100, 850);
    return wrap;
  }

  function renderResult(data) {
    verdict.replaceChildren();
    const fam = "fam-" + data.family;

    // Hero: family colour as a wash and a left rule, class name in text, family
    // named on the chip. Colour never carries the meaning by itself.
    const hero = el("div", "verdict-hero " + fam);
    const top = el("div", "verdict-hero-top");
    const chip = el("span", "fam-chip");
    chip.appendChild(el("span", "swatch"));
    chip.appendChild(el("span", null, data.family_name));
    top.appendChild(chip);
    hero.appendChild(top);
    hero.appendChild(el("div", "verdict-class", data.label));
    hero.appendChild(el("div", "verdict-sub",
      data.is_attack ? "Attack traffic detected" : "Normal traffic — nothing to action"));
    verdict.appendChild(hero);

    const rw = el("div", "ring-wrap");
    rw.appendChild(ring(data.confidence, fam));
    const note = el("div", "ring-note");
    const strong = el("strong", null, "Confidence");
    note.appendChild(strong);
    note.appendChild(el("div", null, "how sure the model is about this class"));
    note.appendChild(el("div", "field-hint", data.model));
    rw.appendChild(note);
    verdict.appendChild(rw);

    const bars = el("div", "bars");
    data.top.forEach(function (row) {
      const rowFam = "fam-" + row.family;
      const barRow = el("div", "bar-row " + rowFam);

      const meta = el("div", "bar-meta");
      const name = el("span", "name");
      name.appendChild(el("span", "bar-dot"));
      name.appendChild(el("span", null, row.label));
      meta.appendChild(name);
      meta.appendChild(el("span", "value", percent(row.confidence)));
      barRow.appendChild(meta);

      const track = el("div", "bar-track");
      const fill = el("div", "bar-fill by-family");
      fill.style.width = "0%";
      track.appendChild(fill);
      barRow.appendChild(track);
      bars.appendChild(barRow);

      nextFrame(function () {
        fill.style.width = Math.max(row.confidence * 100, 0.6) + "%";
      });
    });

    const barsWrap = el("div");
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
