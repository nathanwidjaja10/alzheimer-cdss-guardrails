// Client of the blind human audit. All judgements are sent to the server (audit_log.jsonl);
// localStorage only remembers the last auditor name and is never the source of truth for progress.

// Display labels of the fact groups, narrative fields and hallucination categories
const GROUP_LABELS = {
  prediction_anchor: "Prediksi & Probabilitas",
  spatial_xai_anchor: "Perhatian Spasial / XAI",
  clinical_covariates: "Kovariat Klinis",
  model_provenance: "Provenans Model",
};

const NARR_LABELS = {
  visual_findings: "Temuan Visual",
  multimodal_correlation: "Korelasi Multimodal",
  differential_justification: "Justifikasi Diferensial",
  differential_conclusion: "Kesimpulan Diferensial",
  uncertainty_note: "Catatan Ketidakpastian",
};

const CATEGORY_LABELS = {
  ungrounded_number: "Angka tak berdasar",
  fabricated_structure: "Struktur anatomi fiktif",
  diagnosis_mismatch: "Diagnosis tak cocok",
  ood_contradiction: "Kontradiksi status OOD",
  covariate_caveat_breach: "Kovariat dipakai sbg bukti diagnosis",
  label_or_comparison_error: "Label tertukar / salah perbandingan",
  other: "Lainnya",
};

// UI state: auditor name, anatomy matcher, current case, start time and numbers/terms found in the narrative
const state = {
  auditor: "",
  anatomyRegex: null,
  current: null,
  startTime: 0,
  narrativeMatches: null,
};

// localStorage access that tolerates blocked storage
function safeGetLocal(key) {
  try { return localStorage.getItem(key) || ""; } catch { return ""; }
}
function safeSetLocal(key, val) {
  try { localStorage.setItem(key, val); } catch { /* ignore */ }
}

// Escape text for insertion into HTML
function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

// Canonical form of a number string (drops '%' and trailing zeros)
function normalizeNum(raw) {
  const f = parseFloat(raw.replace("%", ""));
  return Number.isNaN(f) ? raw : String(f);
}

// Canonical form of a fact value for matching against narrative highlights
function normalizeForMatch(raw) {
  const trimmed = String(raw).trim();
  if (/^-?[\d.]+%?$/.test(trimmed)) return normalizeNum(trimmed);
  return trimmed.toLowerCase();
}

// Build the regular expression that matches anatomy terms and numbers
async function loadAnatomyRegex() {
  const res = await fetch("/anatomy_terms.json");
  const terms = await res.json();
  const escaped = terms
    .slice()
    .sort((a, b) => b.length - a.length)
    .map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
  const anatPattern = `(?:${escaped.join("|")})`;
  const numPattern = `-?\\d+(?:\\.\\d+)?%?`;
  state.anatomyRegex = new RegExp(`(${anatPattern})|(${numPattern})`, "gi");
}

// Return HTML of the text with anatomy terms and numbers wrapped in <mark>
function highlightText(text) {
  const regex = state.anatomyRegex;
  regex.lastIndex = 0;
  let out = "";
  let last = 0;
  let m;
  while ((m = regex.exec(text)) !== null) {
    out += escapeHtml(text.slice(last, m.index));
    if (m[1] !== undefined) {
      out += `<mark class="hl-anat" data-kind="anat" data-value="${escapeHtml(m[1].toLowerCase())}">${escapeHtml(m[1])}</mark>`;
    } else {
      out += `<mark class="hl-num" data-kind="num" data-value="${escapeHtml(normalizeNum(m[2]))}">${escapeHtml(m[2])}</mark>`;
    }
    last = m.index + m[0].length;
  }
  out += escapeHtml(text.slice(last));
  return out;
}

// Collect the numbers and anatomy terms found in a text
function extractMatches(text) {
  const regex = state.anatomyRegex;
  regex.lastIndex = 0;
  const numbers = new Set();
  const anat = new Set();
  let m;
  while ((m = regex.exec(text)) !== null) {
    if (m[1] !== undefined) anat.add(m[1].toLowerCase());
    else numbers.add(normalizeNum(m[2]));
  }
  return { numbers, anat };
}

// Render a fact value (recursing into objects and arrays), marking values that appear in the narrative
function renderFactValue(key, val, container) {
  if (val === null || typeof val !== "object") {
    const row = document.createElement("div");
    row.className = "fact-row";
    const k = document.createElement("span");
    k.className = "fact-key";
    k.textContent = key;
    const v = document.createElement("span");
    v.className = "fact-val";
    v.textContent = String(val);
    const normalized = normalizeForMatch(String(val));
    v.dataset.value = normalized;
    const matches = state.narrativeMatches;
    if (matches) {
      if (matches.numbers.has(normalized)) {
        v.classList.add("match-num");
      } else if (matches.anat.has(normalized)) {
        v.classList.add("match-anat");
      }
    }
    row.appendChild(k);
    row.appendChild(v);
    container.appendChild(row);
  } else if (Array.isArray(val)) {
    val.forEach((item, i) => renderFactValue(`${key}[${i}]`, item, container));
  } else {
    Object.entries(val).forEach(([k2, v2]) => renderFactValue(`${key}.${k2}`, v2, container));
  }
}

// Render all fact groups
function renderFacts(facts) {
  const body = document.getElementById("factsBody");
  body.innerHTML = "";
  for (const [groupKey, groupVal] of Object.entries(facts)) {
    const group = document.createElement("div");
    group.className = "fact-group";
    const h3 = document.createElement("h3");
    h3.textContent = GROUP_LABELS[groupKey] || groupKey;
    group.appendChild(h3);
    Object.entries(groupVal).forEach(([k, v]) => renderFactValue(k, v, group));
    body.appendChild(group);
  }
}

// Render the class probabilities sorted in descending order for a quick label-to-number check
function renderRankingCheat(pa) {
  const el = document.getElementById("rankingCheat");
  const entries = Object.entries(pa.relative_probabilities).sort((a, b) => b[1] - a[1]);
  const rows = entries
    .map(([cls, p]) => {
      const isArgmax = cls === pa.argmax_class;
      return `<tr class="${isArgmax ? "argmax" : ""}"><td>${cls}</td><td>${p}</td><td>${isArgmax ? "← argmax_class" : ""}</td></tr>`;
    })
    .join("");
  el.innerHTML = `<strong>Cek cepat label ↔ angka</strong><table>${rows}</table>`;
}

// Render the narrative fields with highlighted terms
function renderNarrative(narrative) {
  const body = document.getElementById("narrativeBody");
  body.innerHTML = "";
  for (const [key, label] of Object.entries(NARR_LABELS)) {
    const val = narrative[key];
    if (val === undefined) continue;
    const div = document.createElement("div");
    div.className = "narr-field";
    const h3 = document.createElement("h3");
    h3.textContent = label;
    const p = document.createElement("p");
    p.innerHTML = highlightText(String(val));
    div.appendChild(h3);
    div.appendChild(p);
    body.appendChild(div);
  }
}

// Render the hallucination category checkboxes
function renderCategoryChecks() {
  const el = document.getElementById("categoryChecks");
  el.innerHTML = "";
  for (const [val, label] of Object.entries(CATEGORY_LABELS)) {
    const wrap = document.createElement("label");
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.value = val;
    wrap.appendChild(cb);
    wrap.appendChild(document.createTextNode(" " + label));
    el.appendChild(wrap);
  }
}

// Render the per-field grounding selects, optionally prefilled from a previous verdict
function renderFieldGrounding(prefill) {
  const el = document.getElementById("fieldGroundingBody");
  el.innerHTML = "";
  for (const [key, label] of Object.entries(NARR_LABELS)) {
    const row = document.createElement("div");
    row.className = "fg-row";
    const span = document.createElement("span");
    span.textContent = label;
    const sel = document.createElement("select");
    sel.className = "fg-select";
    sel.dataset.field = key;
    [["", "-- lewati --"], ["supported", "Didukung"], ["partial", "Sebagian"], ["unsupported", "Tidak didukung"]]
      .forEach(([v, t]) => {
        const opt = document.createElement("option");
        opt.value = v;
        opt.textContent = t;
        sel.appendChild(opt);
      });
    if (prefill && prefill[key]) sel.value = prefill[key];
    row.appendChild(span);
    row.appendChild(sel);
    el.appendChild(row);
  }
}

// Reset the audit form, restoring a previous verdict when there is one
function resetForm(previous) {
  document.getElementById("auditForm").hidden = false;
  document.getElementById("revealPanel").hidden = true;
  document.querySelector('input[name=has_hallucination][value=no]').checked = true;
  document.getElementById("categoryFieldset").hidden = true;
  renderCategoryChecks();
  document.getElementById("otherCategoryNote").value = "";
  document.getElementById("confidenceSelect").value = "medium";
  document.getElementById("notesInput").value = "";
  renderFieldGrounding(previous ? previous.field_grounding : null);

  if (previous) {
    if (previous.has_hallucination) {
      document.querySelector('input[name=has_hallucination][value=yes]').checked = true;
      document.getElementById("categoryFieldset").hidden = false;
      (previous.categories || []).forEach((c) => {
        const cb = document.querySelector(`#categoryChecks input[value="${c}"]`);
        if (cb) cb.checked = true;
      });
      document.getElementById("otherCategoryNote").value = previous.other_category_note || "";
    }
    document.getElementById("confidenceSelect").value = previous.confidence || "medium";
    document.getElementById("notesInput").value = previous.notes || "";
  }
}

// Update the progress bar and text
function updateProgress(progress) {
  const pct = progress.total ? Math.round((progress.done / progress.total) * 100) : 0;
  document.getElementById("progressFill").style.width = pct + "%";
  document.getElementById("progressText").textContent =
    `${progress.done} / ${progress.total} selesai (${pct}%) — ditandai halusinasi: ${progress.flagged}`;
}

// Display one case: facts, narrative, form and progress
function renderCase(data) {
  state.current = data;
  state.startTime = performance.now();
  document.getElementById("app").hidden = false;
  document.getElementById("sessionLabel").textContent =
    `Kasus ${data.order_index + 1} / ${data.total} — session_id: ${data.session_id}`;
  const revisitNote = document.getElementById("revisitNote");
  if (data.previous_verdict) {
    revisitNote.hidden = false;
    revisitNote.textContent = `Sudah dinilai sebelumnya (${data.previous_verdict.timestamp}) — submit ulang akan menimpa.`;
  } else {
    revisitNote.hidden = true;
  }
  const allNumbers = new Set();
  const allAnat = new Set();
  Object.values(data.narrative).forEach((v) => {
    const { numbers, anat } = extractMatches(String(v));
    numbers.forEach((n) => allNumbers.add(n));
    anat.forEach((a) => allAnat.add(a));
  });
  state.narrativeMatches = { numbers: allNumbers, anat: allAnat };

  renderFacts(data.facts);
  renderRankingCheat(data.facts.prediction_anchor);
  renderNarrative(data.narrative);
  resetForm(data.previous_verdict);
  if (data.progress) updateProgress(data.progress);
}

// Display the end state when all cases are completed
function renderFinished(progress) {
  document.getElementById("app").hidden = false;
  document.getElementById("sessionLabel").textContent = "Semua kasus untuk auditor ini sudah dinilai.";
  document.getElementById("revisitNote").hidden = true;
  document.getElementById("factsBody").innerHTML = "";
  document.getElementById("narrativeBody").innerHTML = "";
  document.getElementById("rankingCheat").innerHTML = "";
  document.getElementById("formPanel").querySelector("#auditForm").hidden = true;
  document.getElementById("revealPanel").hidden = true;
  updateProgress(progress);
  state.current = null;
}

// Load the next case the auditor has not completed
async function loadNext() {
  const url = `/api/next?auditor=${encodeURIComponent(state.auditor)}`;
  const res = await fetch(url);
  const data = await res.json();
  if (!res.ok) { alert("Gagal memuat kasus: " + (data.error || res.status)); return; }
  if (data.finished) { renderFinished(data.progress); return; }
  renderCase(data);
}

// Load a specific case, for revisiting
async function loadSession(sessionId) {
  const url = `/api/case?auditor=${encodeURIComponent(state.auditor)}&session_id=${encodeURIComponent(sessionId)}`;
  const res = await fetch(url);
  const data = await res.json();
  if (!res.ok) { alert("Gagal memuat sesi: " + (data.error || res.status)); return; }
  renderCase(data);
}

// After submission, show the automatic guardrail verdict and whether it agrees with the auditor
function showReveal(humanHas) {
  const reveal = state.current.reveal;
  const guardrailFoundIssue = !reveal.guardrail_passed;
  const agree = humanHas === guardrailFoundIssue;
  const el = document.getElementById("revealBody");
  el.innerHTML = `
    <p>Guardrail passed: <strong>${reveal.guardrail_passed}</strong>
       (NCR ${reveal.guardrail_ncr}, sumber: ${reveal.guardrail_source})</p>
    <p>Violation types (otomatis): ${reveal.guardrail_violation_types.length ? reveal.guardrail_violation_types.join(", ") : "(tidak ada)"}</p>
    <p class="${agree ? "agree" : "disagree"}">
      ${agree ? "Sejalan dengan penilaian guardrail otomatis." : "TIDAK sejalan dengan guardrail otomatis — kasus penting untuk dicatat."}
    </p>`;
  document.getElementById("auditForm").hidden = true;
  document.getElementById("revealPanel").hidden = false;
}

// Start: store the auditor name and load the first case
document.getElementById("startBtn").addEventListener("click", async () => {
  const name = document.getElementById("auditorInput").value.trim();
  if (!name) { alert("Isi nama/inisial auditor dulu."); return; }
  state.auditor = name;
  safeSetLocal("audit_auditor_name", name);
  await loadNext();
});

// Jump to a given session
document.getElementById("jumpBtn").addEventListener("click", async () => {
  const sid = document.getElementById("jumpInput").value.trim();
  if (!state.auditor) { alert("Isi nama auditor & tekan Mulai dulu."); return; }
  if (!sid) return;
  await loadSession(sid);
});

// Submit the verdict, then reveal the guardrail result
document.getElementById("auditForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const has = document.querySelector('input[name=has_hallucination]:checked').value === "yes";
  const categories = has
    ? Array.from(document.querySelectorAll("#categoryChecks input:checked")).map((x) => x.value)
    : [];
  const other = document.getElementById("otherCategoryNote").value;
  const confidence = document.getElementById("confidenceSelect").value;
  const notes = document.getElementById("notesInput").value;
  const fieldGrounding = {};
  document.querySelectorAll(".fg-select").forEach((sel) => {
    if (sel.value) fieldGrounding[sel.dataset.field] = sel.value;
  });
  const duration_ms = Math.round(performance.now() - state.startTime);

  const payload = {
    auditor_id: state.auditor,
    session_id: state.current.session_id,
    has_hallucination: has,
    categories,
    other_category_note: other,
    field_grounding: Object.keys(fieldGrounding).length ? fieldGrounding : null,
    confidence,
    notes,
    duration_ms,
  };
  const res = await fetch("/api/submit", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await res.json();
  if (!res.ok) { alert("Gagal submit: " + (data.error || res.status)); return; }
  showReveal(has);
  updateProgress(data.progress);
});

// Show the category choices only when a hallucination is reported
document.getElementById("auditForm").addEventListener("change", (e) => {
  if (e.target.name === "has_hallucination") {
    document.getElementById("categoryFieldset").hidden = e.target.value !== "yes";
  }
});

// Continue to the next case
document.getElementById("nextBtn").addEventListener("click", loadNext);

// Clicking a highlighted term pulses the matching fact values and scrolls to the first one
document.getElementById("narrativeBody").addEventListener("click", (e) => {
  const mark = e.target.closest("mark");
  if (!mark) return;
  const value = mark.dataset.value;
  const candidates = Array.from(document.querySelectorAll(".fact-val")).filter((el) => {
    const fv = el.dataset.value;
    return fv === value || fv.includes(value) || value.includes(fv);
  });
  document.querySelectorAll(".fact-val.pulse").forEach((el) => el.classList.remove("pulse"));
  candidates.forEach((el) => el.classList.add("pulse"));
  if (candidates.length) candidates[0].scrollIntoView({ behavior: "smooth", block: "center" });
});

// Initialisation: load the anatomy matcher and the remembered auditor name
(async function init() {
  await loadAnatomyRegex();
  const remembered = safeGetLocal("audit_auditor_name");
  if (remembered) document.getElementById("auditorInput").value = remembered;
})();
