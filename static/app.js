/* ============================================================
   Semantic Clinical Matching — Web UI Application Logic
   ============================================================ */

// ── Sample job postings ────────────────────────────────────

const SAMPLES = {
    icu: `Title: ICU Registered Nurse
Location: Calgary, Alberta, Canada
Employer: Alberta Health Services

Requirements:
- Active registration with CARNA (College and Association of Registered Nurses of Alberta)
- Minimum 2 years ICU/critical care experience
- BLS and ACLS certification required
- CNCCP(C) certification preferred
- Experience with hemodynamic monitoring and ventilator management
- Strong assessment and critical thinking skills
- Ability to work rotating shifts including nights and weekends`,

    cardiologist: `Title: Staff Cardiologist — Interventional
Location: Toronto, Ontario, Canada
Employer: University Health Network

Requirements:
- Active CPSO registration (College of Physicians and Surgeons of Ontario)
- FRCPC in Internal Medicine and Cardiology
- Fellowship training in Interventional Cardiology
- Minimum 3 years post-fellowship experience
- Expertise in PCI, structural heart interventions, and TAVR
- ACLS certification
- Academic appointment at University of Toronto preferred`,

    pharmacist: `Title: Clinical Pharmacist — Oncology
Location: Vancouver, British Columbia, Canada
Employer: BC Cancer Agency

Requirements:
- Licensed pharmacist with BCPA (College of Pharmacists of BC)
- PharmD or equivalent
- Minimum 2 years oncology pharmacy experience
- Certification in oncology pharmacy (BPS-BCOP) preferred
- Experience with chemotherapy dosing and protocol management
- Knowledge of drug interactions in cancer therapy
- Excellent communication skills for patient counselling`,

    physio: `Title: Registered Physiotherapist — Orthopedics
Location: Ottawa, Ontario, Canada
Employer: The Ottawa Hospital

Requirements:
- Active registration with the College of Physiotherapists of Ontario
- Master's degree in Physiotherapy
- Minimum 1 year experience in orthopedic rehabilitation
- Manual therapy certification (FCAMPT) preferred
- Experience with post-surgical knee and hip rehabilitation
- Knowledge of evidence-based practice
- BLS certification required`,
};

// ── State ──────────────────────────────────────────────────

let currentTab = "ranked";
let lastResult = null;
let availableModels = [];

// ── DOM helpers ────────────────────────────────────────────

const $ = (id) => document.getElementById(id);

function setStatus(text, type = "") {
    const bar = $("statusBar");
    const label = $("statusText");
    bar.style.display = "flex";
    bar.className = "status-bar" + (type ? ` status-bar--${type}` : "");
    label.textContent = text;
}

function hideStatus() {
    $("statusBar").style.display = "none";
}

// ── Model loading ────────────────────────────────────────

function setModelOptions(models, defaultModel) {
    const select = $("llmModel");
    availableModels = models;

    if (!models.length) {
        select.innerHTML = `<option value="">No generative models</option>`;
        select.disabled = true;
        return;
    }

    select.innerHTML = models
        .map((model) => `<option value="${escapeHtml(model)}">${escapeHtml(model)}</option>`)
        .join("");

    const preferredModel = models.includes(defaultModel) ? defaultModel : models[0];
    select.value = preferredModel;
    select.disabled = false;
}

async function loadModels() {
    const select = $("llmModel");
    select.disabled = true;
    select.innerHTML = `<option value="">Loading…</option>`;

    try {
        const resp = await fetch("/models");
        if (!resp.ok) {
            const err = await resp.json().catch(() => ({}));
            throw new Error(err.detail || `HTTP ${resp.status}`);
        }

        const data = await resp.json();
        setModelOptions(data.models || [], data.default_model || "");
    } catch (err) {
        availableModels = [];
        select.innerHTML = `<option value="">Unavailable</option>`;
        select.disabled = true;
        setStatus(`Model list unavailable: ${err.message}`, "error");
    }
}

// ── Sample loading ─────────────────────────────────────────

function loadSample(key) {
    const textarea = $("jobText");
    textarea.value = SAMPLES[key] || "";
    textarea.focus();
    textarea.setSelectionRange(0, 0);
}

// ── Clear ──────────────────────────────────────────────────

function clearAll() {
    $("jobText").value = "";
    $("resultsContent").innerHTML = `
    <div class="results-empty">
      <div class="results-empty__icon">🔍</div>
      <div class="results-empty__text">
        Paste a job posting and click <strong>Match Candidates</strong> to find the best clinical matches.
      </div>
    </div>`;
    $("stageTabs").style.display = "none";
    $("pipelineStats").style.display = "none";
    hideStatus();
    lastResult = null;
}

// ── Ingest ─────────────────────────────────────────────────

async function runIngest() {
    const btn = $("ingestBtn");
    btn.disabled = true;
    btn.textContent = "⏳ Ingesting…";
    setStatus("Embedding resumes and building FAISS index…", "loading");

    try {
        const resp = await fetch("/resumes/ingest", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
        });

        if (!resp.ok) {
            const err = await resp.json().catch(() => ({}));
            throw new Error(err.detail || `HTTP ${resp.status}`);
        }

        const data = await resp.json();
        setStatus(
            `✓ ${data.message}`,
            "success"
        );
    } catch (err) {
        setStatus(`Ingest error: ${err.message}`, "error");
    } finally {
        btn.disabled = false;
        btn.textContent = "📥 Ingest Resumes";
    }
}

// ── Match ──────────────────────────────────────────────────

async function runMatch() {
    const jobText = $("jobText").value.trim();
    if (!jobText) {
        setStatus("Please enter a job posting first", "error");
        return;
    }

    const topK = parseInt($("topK").value) || 5;
    const llmModel = $("llmModel").value;
    const matchBtn = $("matchBtn");

    if (!llmModel) {
        setStatus("No Ollama model selected. Check /models and Ollama server status.", "error");
        return;
    }

    // Show loading
    matchBtn.disabled = true;
    matchBtn.textContent = "⏳ Matching…";
    setStatus("Stage 1: Retrieving candidates via FAISS…", "loading");

    $("stageTabs").style.display = "none";
    $("pipelineStats").style.display = "none";
    $("resultsContent").innerHTML = `
    <div class="spinner-overlay">
      <div class="spinner"></div>
      <div class="spinner-overlay__text">Analyzing candidates…</div>
      <div class="spinner-overlay__sub">Stage 1: FAISS retrieval → Stage 2: LLM reranking</div>
    </div>`;

    try {
        const resp = await fetch("/match", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                job_text: jobText,
                top_k: topK,
                llm_model: llmModel,
            }),
        });

        if (!resp.ok) {
            const err = await resp.json().catch(() => ({}));
            throw new Error(err.detail || `HTTP ${resp.status}`);
        }

        lastResult = await resp.json();
        renderResults(lastResult);
        setStatus(
            `Done — ${lastResult.ranked_candidates.length} candidates evaluated`,
            "success"
        );
    } catch (err) {
        setStatus(`Error: ${err.message}`, "error");
        $("resultsContent").innerHTML = `
      <div class="results-empty">
        <div class="results-empty__icon">⚠️</div>
        <div class="results-empty__text">
          ${escapeHtml(err.message)}<br><br>
          <small>Make sure the server is running and resumes have been ingested.</small>
        </div>
      </div>`;
    } finally {
        matchBtn.disabled = false;
        matchBtn.textContent = "✨ Match Candidates";
    }
}

// ── Render results ─────────────────────────────────────────

function renderResults(data) {
    // Show tabs and stats
    $("stageTabs").style.display = "flex";
    $("pipelineStats").style.display = "grid";

    // Compute stats
    const retrieved = data.retrieval_results.length;
    const passed = data.ranked_candidates.filter(
        (c) => c.status === "PASS"
    ).length;
    const failed = data.ranked_candidates.filter(
        (c) => c.status === "FAIL"
    ).length;

    $("statRetrieved").textContent = retrieved;
    $("statPassed").textContent = passed;
    $("statFailed").textContent = failed;

    // Reset tab state
    currentTab = "ranked";
    $("tabRanked").classList.add("stage-tab--active");
    $("tabRetrieval").classList.remove("stage-tab--active");

    renderRankedCandidates(data.ranked_candidates);
}

function showTab(tab) {
    currentTab = tab;
    $("tabRanked").classList.toggle("stage-tab--active", tab === "ranked");
    $("tabRetrieval").classList.toggle("stage-tab--active", tab === "retrieval");

    if (!lastResult) return;

    if (tab === "ranked") {
        renderRankedCandidates(lastResult.ranked_candidates);
    } else {
        renderRetrievalResults(lastResult.retrieval_results);
    }
}

// ── Render ranked candidates ───────────────────────────────

function renderRankedCandidates(candidates) {
    if (!candidates.length) {
        $("resultsContent").innerHTML = `
      <div class="results-empty">
        <div class="results-empty__icon">📋</div>
        <div class="results-empty__text">No candidates were evaluated.</div>
      </div>`;
        return;
    }

    // Sort: PASS first (by rank), then FAIL
    const sorted = [...candidates].sort((a, b) => {
        if (a.status === "PASS" && b.status !== "PASS") return -1;
        if (a.status !== "PASS" && b.status === "PASS") return 1;
        if (a.rank != null && b.rank != null) return a.rank - b.rank;
        return 0;
    });

    const html = sorted
        .map(
            (c) => `
    <div class="candidate-card" onclick="toggleCard(this)">
      <div class="candidate-card__header">
        <div class="candidate-card__id-group">
          <div class="candidate-card__rank ${c.status === "PASS"
                    ? "candidate-card__rank--pass"
                    : "candidate-card__rank--fail"
                }">
            ${c.status === "PASS" ? c.rank || "—" : "✕"}
          </div>
          <span class="candidate-card__name">${escapeHtml(c.resume_id)}</span>
        </div>
        <span class="candidate-card__badge ${c.status === "PASS"
                    ? "candidate-card__badge--pass"
                    : "candidate-card__badge--fail"
                }">
          ${c.status}
        </span>
      </div>

      ${c.skill_overlaps.length
                    ? `<div class="candidate-card__tags">
              ${c.skill_overlaps
                        .map(
                            (s) =>
                                `<span class="tag tag--skill"><span class="tag__icon">✓</span>${escapeHtml(
                                    s
                                )}</span>`
                        )
                        .join("")}
            </div>`
                    : ""
                }

      ${c.missing_criteria.length
                    ? `<div class="candidate-card__tags">
              ${c.missing_criteria
                        .map(
                            (m) =>
                                `<span class="tag tag--missing"><span class="tag__icon">✗</span>${escapeHtml(
                                    m
                                )}</span>`
                        )
                        .join("")}
            </div>`
                    : ""
                }

      <div class="candidate-card__reasoning">${escapeHtml(c.reasoning)}</div>
      <div class="candidate-card__expand-hint">click to ${"expand"
                }</div>
    </div>`
        )
        .join("");

    $("resultsContent").innerHTML = `<div class="candidate-list">${html}</div>`;
}

// ── Render retrieval results ───────────────────────────────

function renderRetrievalResults(results) {
    if (!results.length) {
        $("resultsContent").innerHTML = `
      <div class="results-empty">
        <div class="results-empty__icon">📋</div>
        <div class="results-empty__text">No retrieval results.</div>
      </div>`;
        return;
    }

    const html = results
        .map(
            (r, i) => `
    <div class="retrieval-row" style="--i: ${i}">
      <span class="retrieval-row__id">${escapeHtml(r.resume_id)}</span>
      <span class="retrieval-row__score ${getScoreClass(r.score)}">
        ${r.score.toFixed(4)}
      </span>
    </div>`
        )
        .join("");

    $("resultsContent").innerHTML = `<div class="retrieval-list">${html}</div>`;
}

// ── Helpers ────────────────────────────────────────────────

function toggleCard(card) {
    card.classList.toggle("candidate-card--expanded");
    const hint = card.querySelector(".candidate-card__expand-hint");
    if (hint) {
        hint.textContent = card.classList.contains("candidate-card--expanded")
            ? "click to collapse"
            : "click to expand";
    }
}

function getScoreClass(score) {
    if (score >= 0.7) return "score--high";
    if (score >= 0.4) return "score--medium";
    return "score--low";
}

function escapeHtml(str) {
    if (!str) return "";
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
}

// ── Keyboard shortcut ──────────────────────────────────────

document.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
        e.preventDefault();
        runMatch();
    }
});

document.addEventListener("DOMContentLoaded", () => {
    loadModels();
});
