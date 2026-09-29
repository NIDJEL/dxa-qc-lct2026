(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const state = { files: [], jobId: null, result: null, entries: [], filter: "all", zoom: 1, x: 0, y: 0 };
  const escapeHTML = (value) => String(value ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
  const percent = (p) => p === null || p === undefined ? "—" : `${(Number(p) * 100).toFixed(1)}%`;
  const views = ["upload", "progress", "results", "detail"];
  function view(name) {
    views.forEach((v) => $(`${v}-view`).classList.toggle("hidden", v !== name));
    window.scrollTo({ top: 0, behavior: "instant" });
  }
  function toast(message) {
    $("toast").textContent = message;
    $("toast").classList.remove("hidden");
    window.setTimeout(() => $("toast").classList.add("hidden"), 6000);
  }
  function getName(file) { return file.webkitRelativePath || file.name; }
  function addFiles(files) {
    const newFiles = Array.from(files).filter((file) => file.name.toLowerCase().endsWith(".dcm"));
    if (newFiles.length !== files.length) toast("Добавлены только файлы .dcm. Формат дополнительно проверяется сервером.");
    const known = new Set(state.files.map((f) => `${getName(f)}|${f.size}|${f.lastModified}`));
    for (const file of newFiles) {
      const key = `${getName(file)}|${file.size}|${file.lastModified}`;
      if (!known.has(key)) state.files.push(file);
      known.add(key);
    }
    if (state.files.length > 512) {
      state.files = state.files.slice(0, 512);
      toast("Максимум 512 файлов на одно задание.");
    }
    renderSelected();
  }
  function renderSelected() {
    $("selected-panel").classList.toggle("hidden", !state.files.length);
    $("selected-count").textContent = `· ${state.files.length}`;
    $("selected-list").innerHTML = state.files.map((f) =>
      `<div class="selected-item"><span>${escapeHTML(getName(f))}</span><span>${(f.size / 1048576).toFixed(2)} МБ</span></div>`).join("");
  }
  $("pick-files").onclick = (e) => { e.stopPropagation(); $("files-input").click(); };
  $("pick-folder").onclick = (e) => { e.stopPropagation(); $("folder-input").click(); };
  $("files-input").onchange = (e) => addFiles(e.target.files);
  $("folder-input").onchange = (e) => addFiles(e.target.files);
  $("dropzone").onclick = (e) => { if (e.target.closest(".pick-actions")) return; $("files-input").click(); };
  $("dropzone").onkeydown = (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); $("files-input").click(); } };
  for (const event of ["dragenter", "dragover"]) $("dropzone").addEventListener(event, (e) => { e.preventDefault(); $("dropzone").classList.add("drag"); });
  for (const event of ["dragleave", "drop"]) $("dropzone").addEventListener(event, (e) => { e.preventDefault(); $("dropzone").classList.remove("drag"); });
  $("dropzone").addEventListener("drop", (e) => addFiles(e.dataTransfer.files));
  $("clear-files").onclick = () => { state.files = []; $("files-input").value = ""; $("folder-input").value = ""; renderSelected(); };
  $("new-analysis").onclick = () => { state.files = []; state.result = null; renderSelected(); view("upload"); };
  $("back-results").onclick = () => view("results");
  async function request(url, options) {
    const response = await fetch(url, options);
    if (!response.ok) {
      let message = `${response.status} ${response.statusText}`;
      try { const body = await response.json(); message = body.detail || message; } catch (_) { /* no JSON */ }
      throw new Error(message);
    }
    return response.json();
  }
  $("start-analysis").onclick = async () => {
    if (!state.files.length) return;
    const total = state.files.reduce((a, f) => a + f.size, 0);
    if (total > 512 * 1048576) return toast("Суммарный размер файлов превышает 512 МБ.");
    const button = $("start-analysis");
    button.disabled = true;
    view("progress");
    $("progress-files").textContent = `${state.files.length} файл(ов) в задании`;
    $("progress-stage").textContent = "Загрузка файлов в локальный сервис";
    try {
      const form = new FormData();
      for (const file of state.files) form.append("files", file, getName(file));
      form.append("singleton_mode", $("singleton-mode").checked ? "true" : "false");
      const job = await request("/api/jobs", { method: "POST", body: form });
      state.jobId = job.job_id;
      poll();
    } catch (error) {
      toast(`Загрузка не выполнена: ${error.message}`);
      view("upload");
    } finally {
      button.disabled = false;
    }
  };
  async function poll() {
    try {
      const job = await request(`/api/jobs/${state.jobId}`);
      $("progress-percent").textContent = `${job.progress}%`;
      $("progress-bar").style.width = `${job.progress}%`;
      $("progress-stage").textContent = job.stage;
      if (job.status === "completed") {
        state.result = await request(`/api/jobs/${state.jobId}/results`);
        renderResults();
        view("results");
      } else if (job.status === "failed") {
        toast(`Ошибка анализа: ${job.error || job.stage}`);
        view("upload");
      } else {
        window.setTimeout(poll, 1100);
      }
    } catch (error) {
      toast(`Нет ответа сервиса: ${error.message}`);
      view("upload");
    }
  }
  function statusFor(row, error) {
    if (error || !row || row.processing_status !== "Success") return { label: "Недоступно", kind: "unknown" };
    if (row.violation_type) return { label: "Обнаружено нарушение", kind: "bad" };
    if (row.quality_class === 1) return { label: "Требует внимания", kind: "warn" };
    return { label: "Модель не отметила нарушений", kind: "good" };
  }
  function regionCode(row) {
    const name = row?.anatomical_region || "";
    return name.includes("позвоночник") ? "spine" : name.includes("бедра") ? "hip" : "unknown";
  }
  function sharedBagCount(row) {
    if (!row || !state.result || state.result.web_grouping?.mode === "experimental_singleton_bag") return 0;
    const key = row.source_study_id;
    if (!key) return 0;
    return (state.result.rows || []).filter((other) =>
      other.processing_status === "Success" && other.source_study_id === key).length;
  }
  function sharedBagText(row) {
    const count = sharedBagCount(row);
    return count > 1 ? `Общий study-level вектор для ${count} снимков · не индивидуальная оценка` : "";
  }
  function renderResults() {
    const result = state.result;
    const summary = result.rows || [];
    const errorByPath = new Map((result.errors || []).map((e) => [e.image_path, e]));
    const entries = [];
    const knownPaths = new Set();
    for (const row of summary) {
      knownPaths.add(row.image_path);
      if (row.internal_image_path) knownPaths.add(row.internal_image_path);
      entries.push({ row, error: errorByPath.get(row.image_path) || errorByPath.get(row.internal_image_path) || null, id: row.web_image_id,
        name: row.relative_path || row.image_uid || "DICOM" });
    }
    for (const error of result.errors || []) {
      if (knownPaths.has(error.image_path)) continue;
      entries.push({ row: null, error, id: null, name: (error.image_path || "").split(/[\\/]/).pop() || "Ошибка DICOM" });
    }
    state.entries = entries;
    const metrics = [
      [summary.length, "Обработано"], [summary.filter((r) => regionCode(r) === "spine").length, "Позвоночник"],
      [summary.filter((r) => regionCode(r) === "hip").length, "Бедро"],
      [summary.filter((r) => r.quality_class === 0).length, "Качество: норма"],
      [summary.filter((r) => r.quality_class === 1 || r.violation_type).length, "Требует внимания"],
      [(result.errors || []).length, "Ошибки"],
    ];
    $("metrics").innerHTML = metrics.map(([v, label]) => `<div class="metric"><strong>${v}</strong><span>${label}</span></div>`).join("");
    const studies = new Map();
    for (const row of summary) {
      const key = row.source_study_id || row.study_uid || "unknown";
      studies.set(key, (studies.get(key) || 0) + 1);
    }
    $("study-list").innerHTML = [...studies].map(([key, count], i) =>
      `<span class="study-chip">Исследование ${i + 1} · ${count} снимк. · ${escapeHTML(key.slice(0, 21))}${key.length > 21 ? "…" : ""}</span>`).join("");
    const warnings = ["Оценки E007 A3 — study-level proxy, отображаемые на снимках; не подтверждённая image-level точность. E023 имеет известные ошибки маршрутизации бедра. Формат CSV ожидает подтверждения организаторов."];
    if (result.web_grouping?.mode === "experimental_singleton_bag") warnings.push("Экспериментальный singleton-bag: отдельный bag на снимок; CSV отключён, результат только предварительный.");
    if (result.web_grouping?.mode === "single_input_bag") warnings.push("Одиночный снимок: результат предварительный; точность singleton-bag не подтверждена, CSV отключён.");
    if ((result.errors || []).length) warnings.push(`${result.errors.length} ошибок/неопределённых маршрутов. Подробности сохранены в JSON.`);
    $("result-warning").textContent = warnings.join(" ");
    $("result-warning").classList.remove("hidden");
    $("export-json").href = `/api/jobs/${state.jobId}/export.json`;
    $("export-csv").href = `/api/jobs/${state.jobId}/export.csv`;
    const csvDisabled = result.web_export?.official_export_eligible === false;
    $("export-csv").classList.toggle("disabled", csvDisabled);
    $("export-csv").setAttribute("aria-disabled", csvDisabled ? "true" : "false");
    renderGallery();
  }
  function visibleEntry(entry) {
    const { row, error } = entry;
    switch (state.filter) {
      case "spine": return regionCode(row) === "spine";
      case "hip": return regionCode(row) === "hip";
      case "good": return row && row.quality_class === 0 && !row.violation_type;
      case "attention": return row && (row.quality_class === 1 || !!row.violation_type);
      case "violations": return row && !!row.violation_type;
      case "errors": return !!error || !row || row.processing_status !== "Success";
      default: return true;
    }
  }
  function renderGallery() {
    const entries = state.entries.filter(visibleEntry);
    $("gallery-empty").classList.toggle("hidden", !!entries.length);
    $("gallery").innerHTML = entries.map((entry) => {
      const status = statusFor(entry.row, entry.error);
      const idx = state.entries.indexOf(entry);
      const region = entry.row?.anatomical_region || "Неизвестная область";
      const image = entry.id ?
        `<img loading="lazy" src="/api/jobs/${state.jobId}/images/${entry.id}" alt="Превью ${escapeHTML(entry.name)}">` :
        `<span class="image-fallback">Просмотр недоступен</span>`;
      return `<button type="button" class="image-card" data-index="${idx}">
        <div class="card-image">${image}<span class="card-status ${status.kind}">${status.label}</span></div>
        <div class="card-body"><span class="card-id">${escapeHTML(entry.name)}</span><h3>${escapeHTML(region)}</h3>
        <p>${escapeHTML(entry.row?.violation_type || entry.error?.error_type || "Модель не отметила нарушений")}${sharedBagText(entry.row) ? `<br>${escapeHTML(sharedBagText(entry.row))}` : ""}</p>
        <div class="card-foot"><span>${entry.row?.quality_class === null || !entry.row ? "Оценка недоступна" : "Quality-head · " + (entry.row.quality_class === 0 ? "норма" : "внимание")}</span><b>↗</b></div></div></button>`;
    }).join("");
    document.querySelectorAll(".image-card").forEach((card) => card.onclick = () => showDetail(Number(card.dataset.index)));
  }
  document.querySelectorAll(".filter").forEach((button) => button.onclick = () => {
    state.filter = button.dataset.filter;
    document.querySelectorAll(".filter").forEach((other) => other.classList.toggle("active", other === button));
    renderGallery();
  });
  const violations = [
    ["spine_positioning", "Некорректная укладка", [0]],
    ["spine_axis", "Отклонение оси позвоночника", [1]],
    ["spine_foreign_object", "Посторонние предметы", [2]],
    ["femur_positioning_rotation", "Некорректная укладка / ротация", [3, 5]],
    ["femur_roi", "Некорректная область интереса", [4, 6]],
  ];
  const headNames = ["spine_positioning", "spine_axis", "spine_foreign_object", "right_hip_positioning_rotation", "right_hip_roi", "left_hip_positioning_rotation", "left_hip_roi", "quality_spine", "quality_right_hip", "quality_left_hip"];
  function showDetail(index) {
    const entry = state.entries[index], row = entry.row;
    const region = regionCode(row), status = statusFor(row, entry.error);
    $("detail-name").textContent = entry.name;
    $("detail-counter").textContent = `${index + 1} / ${state.entries.length}`;
    const image = $("detail-image");
    image.classList.toggle("hidden", !entry.id);
    $("viewer-error").classList.toggle("hidden", !!entry.id);
    if (entry.id) {
      image.src = `/api/jobs/${state.jobId}/images/${entry.id}`;
      image.onerror = () => { image.classList.add("hidden"); $("viewer-error").classList.remove("hidden"); };
    }
    resetZoom();
    const probs = row?.study_proxy_probabilities || [];
    const thresholds = state.result.provenance?.thresholds || [];
    const applicable = region === "spine" ? violations.slice(0, 3) : region === "hip" ? violations.slice(3) : [];
    const violationHTML = applicable.length ? applicable.map(([, label, indices]) => {
      const probability = Math.max(...indices.map((i) => Number(probs[i] ?? 0)));
      const flagged = indices.some((i) => Number(probs[i]) >= Number(thresholds[i]));
      return `<div class="violation"><div class="violation-top"><strong>${label}</strong><span>${percent(probability)}</span></div>
        <div class="bar ${flagged ? "flag" : ""}"><span style="width:${Math.min(100, probability * 100)}%"></span></div>
        <small>${flagged ? "Превышен сохранённый порог" : "Ниже сохранённого порога"}${indices.length > 1 ? " · сторона бедра неизвестна, максимум голов" : ""}</small></div>`;
    }).join("") : '<p class="no-violation">При неизвестной анатомической области нарушения не оцениваются.</p>';
    const technical = {
      head_probabilities: Object.fromEntries(headNames.map((name, i) => [name, probs[i] ?? null])),
      thresholds: Object.fromEntries(headNames.map((name, i) => [name, thresholds[i] ?? null])),
      temperatures: state.result.provenance?.temperatures || null,
      checkpoint: state.result.provenance?.checkpoint || null,
      checkpoint_sha256: state.result.provenance?.checkpoint_sha256 || null,
      router_sha256: state.result.provenance?.router_sha256 || null,
      router_status: row?.routing_status || null,
      router_p_spine: row?.router_p_spine ?? null,
      grouping: state.result.web_grouping,
      processing_seconds: row?.time_of_processing ?? null,
      error: entry.error || null,
    };
    $("detail-panel").innerHTML = `<span class="panel-label">АНАТОМИЧЕСКАЯ ОБЛАСТЬ</span>
      <div class="panel-region">${escapeHTML(row?.anatomical_region || "Не определена")}</div>
      <div class="panel-quality ${status.kind}"><strong>${escapeHTML(status.label)}</strong>
      <small>${row && row.quality_class !== null ? "Отдельная quality-head: " + percent(row.quality_prob) : "Оценка качества недоступна"} · ${escapeHTML(row?.processing_status || "Failure")}</small></div>
      ${sharedBagText(row) ? `<p class="no-violation">${escapeHTML(sharedBagText(row))}. Вероятности повторяются у снимков одной study-bag.</p>` : ""}
      <div class="panel-section"><h3>Проверки нарушений</h3>${violationHTML}
      ${row && !row.violation_type && region !== "unknown" ? '<p class="no-violation">Модель не отметила нарушений по сохранённым порогам.</p>' : ""}</div>
      <details class="technical"><summary>Техническая информация</summary>
      <p>Вероятность router не является клинически откалиброванной уверенностью. Study-level proxy повторяется у снимков bag; официальная image-level точность не установлена.</p>
      <pre>${escapeHTML(JSON.stringify(technical, null, 2))}</pre></details>`;
    view("detail");
  }
  function transform() { $("detail-image").style.transform = `translate(${state.x}px, ${state.y}px) scale(${state.zoom})`; }
  function resetZoom() { state.zoom = 1; state.x = state.y = 0; transform(); }
  $("zoom-reset").onclick = resetZoom;
  $("zoom-in").onclick = () => { state.zoom = Math.min(8, state.zoom * 1.25); transform(); };
  $("zoom-out").onclick = () => { state.zoom = Math.max(0.3, state.zoom / 1.25); transform(); };
  $("viewer").onwheel = (e) => { e.preventDefault(); state.zoom = Math.max(0.3, Math.min(8, state.zoom * (e.deltaY < 0 ? 1.12 : .89))); transform(); };
  let drag = null;
  $("viewer").onpointerdown = (e) => { drag = { x: e.clientX - state.x, y: e.clientY - state.y }; $("viewer").setPointerCapture(e.pointerId); };
  $("viewer").onpointermove = (e) => { if (drag) { state.x = e.clientX - drag.x; state.y = e.clientY - drag.y; transform(); } };
  $("viewer").onpointerup = () => { drag = null; };
})();
