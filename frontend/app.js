/**
 * MedVisionAI — Unified Clinical Frontend Controller
 * Connects Brain Tumor MRI Studio, Colorectal H&E Studio, Medical AI Assistant,
 * Specialist Appointments & Triage Queue, and Comparative Study Benchmarks.
 */

const state = {
  activeTab: "overview",
  brainSamples: [],
  crcSamples: [],
  selectedBrainSample: "glioma_axial_t1.jpg",
  selectedCrcSample: "test_colorectal_adenocarcinoma_epithelium_0.png",
  uploadedBrainFile: null,
  uploadedCrcFile: null,
  brainViewMode: "overlay", // 'overlay' | 'probability' | 'saliency' | 'binary_mask'
  crcViewMode: "overlay",   // 'overlay' | 'heatmap_only'
  lastBrainResult: null,
  lastCrcResult: null,
  analyzedCases: [], // History of all analyzed scans across both modalities
  selectedSpecialistId: "neuro-oncology",
  specialists: [],
  appointments: [],
  benchmarks: null,
  chatHistory: [
    {
      role: "assistant",
      content:
        "### Welcome to the MedVisionAI Clinical Assistant\n\nI am scoped exclusively to medicine, pathology, and neuro-oncology. You can:\n- **Attach any Brain Tumor MRI or Colorectal H&E scan** from the dropdown above to get a structured breakdown of model predictions, U-Net lesion segmentation, classical ML agreement, and ViT Token Grad-CAM heatmaps.\n- **Ask clinical questions** about glioma, meningioma, pituitary adenoma, colorectal adenocarcinoma, cancer-associated stroma, or triage guidelines.",
      provider: "MedVisionAI Clinical Intelligence Engine",
    },
  ],
};

function switchTab(tabName) {
  state.activeTab = tabName;
  const tabs = ["overview", "brain", "colorectal", "assistant", "appointments", "benchmarks"];
  tabs.forEach((t) => {
    const viewEl = document.getElementById(`view-${t}`);
    const btnEl = document.getElementById(`tab-btn-${t}`);
    if (viewEl) viewEl.style.display = t === tabName ? "block" : "none";
    if (btnEl) {
      btnEl.classList.toggle("active", t === tabName);
      btnEl.setAttribute("aria-selected", t === tabName ? "true" : "false");
    }
  });
  window.scrollTo({ top: 0, behavior: "smooth" });

  if (tabName === "brain" && !state.lastBrainResult) {
    runBrainAnalysis();
  } else if (tabName === "colorectal" && !state.lastCrcResult) {
    runCrcAnalysis();
  }
}

function openSampleDirect(modality, sampleId) {
  if (modality === "brain") {
    state.selectedBrainSample = sampleId;
    state.uploadedBrainFile = null;
    switchTab("brain");
    renderBrainSamples();
    runBrainAnalysis();
  } else {
    state.selectedCrcSample = sampleId;
    state.uploadedCrcFile = null;
    switchTab("colorectal");
    renderCrcSamples();
    runCrcAnalysis();
  }
}

async function initApp() {
  renderChatMessages();
  try {
    const [samplesRes, apptsRes, benchRes] = await Promise.all([
      fetch("/api/samples").then((r) => r.json()),
      fetch("/api/appointments").then((r) => r.json()),
      fetch("/api/benchmarks").then((r) => r.json()),
    ]);

    state.brainSamples = samplesRes.brain_mri || [];
    state.crcSamples = samplesRes.colorectal || [];
    renderBrainSamples();
    renderCrcSamples();

    state.specialists = apptsRes.specialists || [];
    state.appointments = apptsRes.appointments || [];
    renderSpecialists();
    renderAppointmentsQueue();

    state.benchmarks = benchRes;
    renderBenchmarks();

    await runBrainAnalysis(true);
    await runCrcAnalysis(true);
  } catch (err) {
    console.error("Failed initializing MedVisionAI:", err);
  }
}

function renderBrainSamples() {
  const container = document.getElementById("brain-samples-container");
  if (!container) return;
  container.innerHTML = state.brainSamples
    .map((s) => {
      const isActive = !state.uploadedBrainFile && state.selectedBrainSample === s.id;
      return `
        <button type="button" class="sample-pill-card ${isActive ? "active" : ""}" onclick="selectBrainSample('${s.id}')">
          <div class="sample-thumb-circle">
            <img src="${s.url}" alt="${s.title}" loading="lazy" />
          </div>
          <div>
            <div class="sample-card-title">${s.title}</div>
            <div class="sample-card-sub">${s.plane.toUpperCase()}</div>
          </div>
        </button>
      `;
    })
    .join("");
}

function selectBrainSample(sampleId) {
  state.selectedBrainSample = sampleId;
  state.uploadedBrainFile = null;
  const label = document.getElementById("brain-upload-label");
  if (label) label.textContent = "Click to upload a Brain MRI slice";
  renderBrainSamples();
  runBrainAnalysis();
}

function handleBrainFileUpload(event) {
  const file = event.target.files?.[0];
  if (!file) return;
  state.uploadedBrainFile = file;
  const label = document.getElementById("brain-upload-label");
  if (label) label.textContent = `Uploaded: ${file.name}`;
  renderBrainSamples();
  runBrainAnalysis();
}

async function runBrainAnalysis(silent = false) {
  const btn = document.getElementById("brain-run-btn");
  const panel = document.getElementById("brain-results-panel");
  if (btn && !silent) {
    btn.disabled = true;
    btn.innerHTML = `<span>Analyzing Brain MRI...</span>`;
  }

  try {
    const fd = new FormData();
    if (state.uploadedBrainFile) {
      fd.append("file", state.uploadedBrainFile);
    } else {
      fd.append("sample_id", state.selectedBrainSample);
    }
    const plane = document.getElementById("brain-plane-select")?.value || "auto";
    const backbone = document.getElementById("brain-backbone-select")?.value || "ensemble";
    const thresh = document.getElementById("brain-thresh-input")?.value || "0.35";
    const tta = document.getElementById("brain-tta-check")?.checked ?? true;

    fd.append("plane", plane);
    fd.append("backbone", backbone);
    fd.append("segmentation_threshold", thresh);
    fd.append("use_tta", tta ? "true" : "false");

    const resp = await fetch("/api/analyze/brain", { method: "POST", body: fd });
    if (!resp.ok) throw new Error(`Analysis failed (${resp.status})`);
    const data = await resp.json();
    state.lastBrainResult = data;
    recordAnalyzedCase(data);
    renderBrainResults(data);
  } catch (err) {
    if (panel) {
      panel.innerHTML = `<div class="white-card"><p style="color: var(--color-ember);">Error analyzing Brain MRI: ${err.message}</p></div>`;
    }
  } finally {
    if (btn && !silent) {
      btn.disabled = false;
      btn.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"/></svg><span>Analyze Brain MRI Now</span>`;
    }
  }
}

function setBrainViewMode(mode) {
  state.brainViewMode = mode;
  if (state.lastBrainResult) renderBrainResults(state.lastBrainResult);
}

function renderBrainResults(r) {
  const panel = document.getElementById("brain-results-panel");
  if (!panel || !r) return;

  const cls = r.classification;
  const seg = r.segmentation;
  const risk = r.risk_summary;
  const classical = r.classical_support;
  const isNoTumor = cls.predicted_class === "no_tumor";

  const riskBadgeClass =
    risk.risk_level === "high" ? "ember" : risk.risk_level === "medium" ? "petal" : "forest";

  const activeOverlayB64 = r.preview_images[state.brainViewMode] || r.preview_images.overlay;
  const viewModeLabels = {
    overlay: "U-Net Lesion Contour Overlay",
    probability: "U-Net Spatial Probability Map",
    saliency: "CNN Input-Gradient Saliency Map",
    binary_mask: "Binary Segmentation Mask",
  };

  const probRowsHtml = Object.entries(cls.probabilities)
    .sort((a, b) => b[1] - a[1])
    .map(([lbl, val]) => {
      const pct = (val * 100).toFixed(2);
      const fillClass = lbl === "no_tumor" ? "forest" : lbl === cls.predicted_class ? "" : "";
      return `
        <div class="prob-bar-row">
          <div class="prob-bar-header">
            <span style="font-weight: 500; text-transform: capitalize;">${lbl.replace("_", " ")}</span>
            <span>${pct}%</span>
          </div>
          <div class="prob-bar-track">
            <div class="prob-bar-fill ${fillClass}" style="width: ${Math.max(2, val * 100)}%;"></div>
          </div>
        </div>
      `;
    })
    .join("");

  const backbonesRowsHtml = Object.entries(cls.all_backbones || {})
    .map(([key, b]) => {
      const isSelected = key === (cls.strategy_name === "efficientnet_b0_fold_ensemble" ? "ensemble" : cls.strategy_name);
      return `
        <tr style="${isSelected ? "background-color: rgba(72, 101, 255, 0.06);" : ""}">
          <td><strong>${b.title}</strong> ${isSelected ? '<span class="info-pill blue" style="padding:2px 8px;font-size:10px;">Active</span>' : ""}</td>
          <td style="text-transform: capitalize; font-weight: 500;">${b.predicted_class.replace("_", " ")}</td>
          <td>${(b.probability * 100).toFixed(2)}%</td>
          <td>${(b.probabilities.glioma * 100).toFixed(1)}% / ${(b.probabilities.meningioma * 100).toFixed(1)}% / ${(b.probabilities.pituitary * 100).toFixed(1)}%</td>
        </tr>
      `;
    })
    .join("");

  const classicalRowsHtml = (classical.all_classical_models || [])
    .map(
      (m) => `
      <tr>
        <td><strong>${m.model_name}</strong></td>
        <td>
          <span class="info-pill ${m.predicted_binary_label === "tumor" ? "ember" : "forest"}" style="padding: 3px 10px; font-size: 11px;">
            ${m.predicted_binary_label.replace("_", " ").toUpperCase()}
          </span>
        </td>
        <td>${(m.tumor_probability * 100).toFixed(2)}%</td>
      </tr>
    `
    )
    .join("");

  const featHtml = Object.entries(r.engineered_features || {})
    .map(
      ([k, v]) => `
      <div style="background: var(--color-cream); padding: 12px 14px; border-radius: 12px;">
        <div style="font-size: 11px; color: var(--color-stone); text-transform: uppercase;">${k.replaceAll("_", " ")}</div>
        <div class="serif-heading" style="font-size: 20px; margin-top: 2px;">${v}</div>
      </div>
    `
    )
    .join("");

  panel.innerHTML = `
    <!-- Top Summary Banner -->
    <div class="white-card" style="display: flex; flex-direction: column; gap: 20px;">
      <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 16px;">
        <div>
          <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
            <span class="info-pill ${isNoTumor ? "forest" : "ember"}">${r.finding.toUpperCase()}</span>
            <span class="info-pill blue">PLANE: ${r.plane.toUpperCase()}</span>
            <span class="info-pill">${r.image_size[0]}×${r.image_size[1]} PX</span>
          </div>
          <h2 class="serif-heading" style="font-size: var(--text-heading);">
            Predicted Pathology: <span style="text-transform: capitalize;">${cls.predicted_class.replace("_", " ")}</span> (${(cls.probability * 100).toFixed(2)}%)
          </h2>
          <p style="font-size: var(--text-body-sm); color: var(--color-stone); margin-top: 4px;">
            Evaluated with <strong>${cls.selected_model_title}</strong> · TTA: ${cls.tta_used ? "Enabled" : "Disabled"} · Source: <code>${r.filename}</code>
          </p>
        </div>

        <div style="display: flex; gap: 10px; flex-wrap: wrap;">
          <button class="btn-primary" onclick="discussCaseInAssistant()">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
            <span>Discuss in AI Assistant</span>
          </button>
          <button class="btn-outline" onclick="referCaseToSpecialist()">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="4" width="18" height="18" rx="2" ry="2"/><line x1="16" y1="2" x2="16" y2="6"/><line x1="8" y1="2" x2="8" y2="6"/><line x1="3" y1="10" x2="21" y2="10"/></svg>
            <span>Refer to Neuro-Oncology</span>
          </button>
        </div>
      </div>

      <!-- 4 Key Clinical KPI Cards -->
      <div class="grid-4">
        <div style="background: var(--color-cream); border-radius: 16px; padding: 18px;">
          <div style="font-size: 12px; color: var(--color-stone);">CNN Confidence</div>
          <div class="serif-heading" style="font-size: 28px; margin-top: 4px;">${(cls.probability * 100).toFixed(1)}%</div>
          <div style="font-size: 12px; color: var(--color-stone); margin-top: 2px;">${cls.selected_model_title}</div>
        </div>
        <div style="background: var(--color-cream); border-radius: 16px; padding: 18px;">
          <div style="font-size: 12px; color: var(--color-stone);">Segmented Burden</div>
          <div class="serif-heading" style="font-size: 28px; margin-top: 4px;">${seg.tumor_area_percent.toFixed(2)}%</div>
          <div style="font-size: 12px; color: var(--color-stone); margin-top: 2px;">Size: ${risk.tumor_size_category.toUpperCase()} (${seg.tumor_area_px} px)</div>
        </div>
        <div style="background: var(--color-cream); border-radius: 16px; padding: 18px;">
          <div style="font-size: 12px; color: var(--color-stone);">Classical RF Support</div>
          <div class="serif-heading" style="font-size: 28px; margin-top: 4px;">${(classical.tumor_probability * 100).toFixed(1)}%</div>
          <div style="font-size: 12px; color: var(--color-stone); margin-top: 2px;">Binary: ${classical.predicted_binary_label.toUpperCase()}</div>
        </div>
        <div style="background: var(--color-cream); border-radius: 16px; padding: 18px;">
          <div style="font-size: 12px; color: var(--color-stone);">Clinical Risk Score</div>
          <div style="display: flex; align-items: baseline; gap: 8px; margin-top: 4px;">
            <span class="serif-heading" style="font-size: 28px;">${risk.risk_score}/100</span>
            <span class="info-pill ${riskBadgeClass}" style="padding: 2px 10px; font-size: 10px;">${risk.risk_level.toUpperCase()}</span>
          </div>
          <div style="font-size: 12px; color: var(--color-stone); margin-top: 2px;">${risk.suggested_priority}</div>
        </div>
      </div>
    </div>

    <!-- Visual Output Stage & Class Probabilities -->
    <div class="grid-2" style="grid-template-columns: 1.35fr 1fr; align-items: start;">
      <div class="white-card" style="display: flex; flex-direction: column; gap: 16px;">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
          <h3 style="font-size: var(--text-heading-sm);">Visual Segmentation &amp; Saliency</h3>
          <div style="display: flex; gap: 6px; flex-wrap: wrap;">
            <button class="nav-pill-btn ${state.brainViewMode === "overlay" ? "active" : ""}" style="padding: 6px 12px; min-height: 34px; font-size: 12px;" onclick="setBrainViewMode('overlay')">Contour</button>
            <button class="nav-pill-btn ${state.brainViewMode === "probability" ? "active" : ""}" style="padding: 6px 12px; min-height: 34px; font-size: 12px;" onclick="setBrainViewMode('probability')">Heatmap</button>
            <button class="nav-pill-btn ${state.brainViewMode === "saliency" ? "active" : ""}" style="padding: 6px 12px; min-height: 34px; font-size: 12px;" onclick="setBrainViewMode('saliency')">Saliency</button>
            <button class="nav-pill-btn ${state.brainViewMode === "binary_mask" ? "active" : ""}" style="padding: 6px 12px; min-height: 34px; font-size: 12px;" onclick="setBrainViewMode('binary_mask')">Mask</button>
          </div>
        </div>

        <div class="image-viewer-stage">
          <div class="image-comparison-pair">
            <div class="image-frame-box">
              <div style="font-size: 12px; font-weight: 500; color: var(--color-stone);">Source Brain MRI (${r.plane})</div>
              <div class="image-aspect-wrapper">
                <img src="data:image/png;base64,${r.preview_images.original}" alt="Original Brain MRI" />
              </div>
            </div>
            <div class="image-frame-box">
              <div style="font-size: 12px; font-weight: 500; color: var(--color-primary-indigo);">${viewModeLabels[state.brainViewMode]}</div>
              <div class="image-aspect-wrapper">
                <img src="data:image/png;base64,${activeOverlayB64}" alt="${viewModeLabels[state.brainViewMode]}" />
              </div>
            </div>
          </div>
        </div>

        <div style="background: var(--color-cream); border-radius: 12px; padding: 14px 16px; font-size: 13px; color: var(--color-deep-ink);">
          <strong>Clinical Risk Synthesis:</strong> ${risk.explanation}
        </div>
      </div>

      <!-- Class Probabilities & Morphological Contour Metrics -->
      <div class="white-card" style="display: flex; flex-direction: column; gap: 20px;">
        <div>
          <h3 style="font-size: var(--text-heading-sm); margin-bottom: 4px;">4-Class Probability Profile</h3>
          <p style="font-size: 12px; color: var(--color-stone); margin-bottom: 14px;">Calibrated softmax output across BRISC 2025 categories</p>
          <div class="prob-bar-list">${probRowsHtml}</div>
        </div>

        <div style="padding-top: 16px; border-top: 1px solid rgba(8,3,49,0.08);">
          <h4 style="font-size: 18px; margin-bottom: 10px;">U-Net Morphological Metrics</h4>
          <div class="grid-2" style="gap: 10px; font-size: 13px;">
            <div style="background: var(--color-cream); padding: 10px 12px; border-radius: 10px;">
              <span style="color: var(--color-stone); display: block; font-size: 11px;">Bounding Box Area</span>
              <strong>${seg.bbox_area} px²</strong>
            </div>
            <div style="background: var(--color-cream); padding: 10px 12px; border-radius: 10px;">
              <span style="color: var(--color-stone); display: block; font-size: 11px;">Lesion Compactness</span>
              <strong>${seg.compactness !== null ? seg.compactness : "N/A"}</strong>
            </div>
            <div style="background: var(--color-cream); padding: 10px 12px; border-radius: 10px;">
              <span style="color: var(--color-stone); display: block; font-size: 11px;">Intensity Contrast</span>
              <strong>${risk.intensity_contrast}</strong>
            </div>
            <div style="background: var(--color-cream); padding: 10px 12px; border-radius: 10px;">
              <span style="color: var(--color-stone); display: block; font-size: 11px;">Lesion Centroid (X, Y)</span>
              <strong>${seg.centroid[0] !== null ? `${seg.centroid[0]}, ${seg.centroid[1]}` : "None"}</strong>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Multi-Model Comparison Tables: 4 Deep CNN Backbones + 6 Classical ML Models -->
    <div class="grid-2">
      <div class="white-card">
        <h3 style="font-size: var(--text-heading-sm); margin-bottom: 6px;">Deep CNN Backbone Consensus</h3>
        <p style="font-size: 12px; color: var(--color-stone); margin-bottom: 14px;">Live side-by-side inference across all 4 trained PyTorch checkpoints</p>
        <div class="clinical-table-wrapper">
          <table class="clinical-table">
            <thead>
              <tr>
                <th>Architecture</th>
                <th>Prediction</th>
                <th>Confidence</th>
                <th>G / M / P (%)</th>
              </tr>
            </thead>
            <tbody>${backbonesRowsHtml}</tbody>
          </table>
        </div>
      </div>

      <div class="white-card">
        <h3 style="font-size: var(--text-heading-sm); margin-bottom: 6px;">Classical ML Radiomic Support</h3>
        <p style="font-size: 12px; color: var(--color-stone); margin-bottom: 14px;">Live predictions from all 6 trained scikit-learn pipelines on 7 engineered features</p>
        <div class="clinical-table-wrapper">
          <table class="clinical-table">
            <thead>
              <tr>
                <th>Classical Model</th>
                <th>Verdict</th>
                <th>Tumor Prob.</th>
              </tr>
            </thead>
            <tbody>${classicalRowsHtml}</tbody>
          </table>
        </div>
      </div>
    </div>

    <!-- Engineered Radiomic Features Bar -->
    <div class="white-card">
      <h3 style="font-size: var(--text-heading-sm); margin-bottom: 12px;">7 Extracted Radiomic &amp; Image Quality Features</h3>
      <div class="grid-4" style="grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px;">
        ${featHtml}
      </div>
    </div>
  `;
}

function renderCrcSamples() {
  const container = document.getElementById("crc-samples-container");
  if (!container) return;
  container.innerHTML = state.crcSamples
    .map((s) => {
      const isActive = !state.uploadedCrcFile && state.selectedCrcSample === s.id;
      return `
        <button type="button" class="sample-pill-card ${isActive ? "active" : ""}" onclick="selectCrcSample('${s.id}')">
          <div class="sample-thumb-circle">
            <img src="${s.url}" alt="${s.title}" loading="lazy" />
          </div>
          <div>
            <div class="sample-card-title">${s.title}</div>
            <div class="sample-card-sub">${s.category.toUpperCase()}</div>
          </div>
        </button>
      `;
    })
    .join("");
}

function selectCrcSample(sampleId) {
  state.selectedCrcSample = sampleId;
  state.uploadedCrcFile = null;
  const label = document.getElementById("crc-upload-label");
  if (label) label.textContent = "Click to upload an H&E tissue image";
  renderCrcSamples();
  runCrcAnalysis();
}

function handleCrcFileUpload(event) {
  const file = event.target.files?.[0];
  if (!file) return;
  state.uploadedCrcFile = file;
  const label = document.getElementById("crc-upload-label");
  if (label) label.textContent = `Uploaded: ${file.name}`;
  renderCrcSamples();
  runCrcAnalysis();
}

async function runCrcAnalysis(silent = false) {
  const btn = document.getElementById("crc-run-btn");
  const panel = document.getElementById("crc-results-panel");
  if (btn && !silent) {
    btn.disabled = true;
    btn.innerHTML = `<span>Analyzing H&amp;E Patch...</span>`;
  }

  try {
    const fd = new FormData();
    if (state.uploadedCrcFile) {
      fd.append("file", state.uploadedCrcFile);
    } else {
      fd.append("sample_id", state.selectedCrcSample);
    }
    const mode = document.getElementById("crc-mode-select")?.value || "Auto";
    const tile = document.getElementById("crc-tile-select")?.value || "Auto";
    const alpha = document.getElementById("crc-alpha-input")?.value || "0.50";
    const frac = document.getElementById("crc-frac-input")?.value || "0.10";
    const force = document.getElementById("crc-force-check")?.checked ?? false;

    fd.append("mode", mode);
    fd.append("tile_size", tile);
    fd.append("heatmap_alpha", alpha);
    fd.append("min_tumor_fraction", frac);
    fd.append("force_out_of_scope", force ? "true" : "false");

    const resp = await fetch("/api/analyze/colorectal", { method: "POST", body: fd });
    if (!resp.ok) throw new Error(`Histopathology analysis failed (${resp.status})`);
    const data = await resp.json();
    state.lastCrcResult = data;
    recordAnalyzedCase(data);
    renderCrcResults(data);
  } catch (err) {
    if (panel) {
      panel.innerHTML = `<div class="white-card"><p style="color: var(--color-ember);">Error analyzing H&E image: ${err.message}</p></div>`;
    }
  } finally {
    if (btn && !silent) {
      btn.disabled = false;
      btn.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"/></svg><span>Analyze H&amp;E Tissue Now</span>`;
    }
  }
}

function setCrcViewMode(mode) {
  state.crcViewMode = mode;
  if (state.lastCrcResult) renderCrcResults(state.lastCrcResult);
}

function renderCrcResults(r) {
  const panel = document.getElementById("crc-results-panel");
  if (!panel || !r) return;

  const isTumor = r.finding === "Tumour tissue detected";
  const isNotAssessed = r.finding === "Not assessed";
  const badgeClass = isTumor ? "ember" : isNotAssessed ? "petal" : "forest";

  const probsEntries = Object.entries(r.class_probabilities || {}).sort((a, b) => b[1] - a[1]);
  const probsHtml = probsEntries.length
    ? probsEntries
        .map(([name, val]) => {
          const pct = (val * 100).toFixed(2);
          const isTumClass = name.includes("adenocarcinoma");
          return `
            <div class="prob-bar-row">
              <div class="prob-bar-header">
                <span style="font-weight: ${isTumClass ? "500" : "400"};">${name} ${isTumClass ? "(Tumour)" : ""}</span>
                <span>${pct}%</span>
              </div>
              <div class="prob-bar-track">
                <div class="prob-bar-fill ${isTumClass ? "ember" : ""}" style="width: ${Math.max(1.5, val * 100)}%;"></div>
              </div>
            </div>
          `;
        })
        .join("")
    : `<p style="font-size: 13px; color: var(--color-stone);">Out-of-domain image: enable "Force analysis" in the left panel if you wish to run the classifier anyway.</p>`;

  const reasonsHtml = (r.confidence_reasons || [])
    .map((msg) => `<li style="margin-bottom: 6px;">${msg}</li>`)
    .join("");

  const activeRightImg =
    state.crcViewMode === "heatmap_only" && r.preview_images.heatmap_only
      ? r.preview_images.heatmap_only
      : r.preview_images.overlay || r.preview_images.original;

  const tilesCardHtml = r.tiles
    ? `
      <div class="white-card">
        <h3 style="font-size: var(--text-heading-sm); margin-bottom: 8px;">Multi-Patch Tile Scan Summary</h3>
        <p style="font-size: 13px; color: var(--color-stone); margin-bottom: 16px;">${r.tiles.rule}</p>
        <div class="grid-4">
          <div style="background: var(--color-cream); padding: 14px; border-radius: 12px;">
            <div style="font-size: 11px; color: var(--color-stone);">Total Grid Tiles</div>
            <div class="serif-heading" style="font-size: 24px;">${r.tiles.n_tiles} (${r.tiles.tile_size_px}px)</div>
          </div>
          <div style="background: var(--color-cream); padding: 14px; border-radius: 12px;">
            <div style="font-size: 11px; color: var(--color-stone);">Stained Tissue Tiles</div>
            <div class="serif-heading" style="font-size: 24px;">${r.tiles.n_tissue_tiles}</div>
          </div>
          <div style="background: var(--color-cream); padding: 14px; border-radius: 12px;">
            <div style="font-size: 11px; color: var(--color-stone);">Flagged Tumour Tiles</div>
            <div class="serif-heading" style="font-size: 24px; color: var(--color-ember);">${r.tiles.n_tumor_tiles}</div>
          </div>
          <div style="background: var(--color-cream); padding: 14px; border-radius: 12px;">
            <div style="font-size: 11px; color: var(--color-stone);">Tumour Tile Fraction</div>
            <div class="serif-heading" style="font-size: 24px;">${(r.tiles.tumor_fraction * 100).toFixed(1)}%</div>
          </div>
        </div>
      </div>
    `
    : "";

  panel.innerHTML = `
    <div class="white-card" style="display: flex; flex-direction: column; gap: 20px;">
      <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 16px;">
        <div>
          <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
            <span class="info-pill ${badgeClass}">${r.finding.toUpperCase()}</span>
            <span class="info-pill blue">CONFIDENCE: ${r.confidence.toUpperCase()}</span>
            <span class="info-pill">${r.analysis_mode.toUpperCase()}</span>
          </div>
          <h2 class="serif-heading" style="font-size: var(--text-heading);">
            ${r.predicted_class ? `${r.predicted_class} (${(r.predicted_probability * 100).toFixed(2)}%)` : "Image Not Assessed (Outside H&E Domain)"}
          </h2>
          <p style="font-size: var(--text-body-sm); color: var(--color-stone); margin-top: 4px;">
            Backbone: <strong>${r.model_info.backbone}</strong> · File: <code>${r.filename}</code> (${r.image_size[0]}×${r.image_size[1]} px)
          </p>
        </div>

        <div style="display: flex; gap: 10px; flex-wrap: wrap;">
          <button class="btn-primary" onclick="discussCaseInAssistant()">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
            <span>Discuss in AI Assistant</span>
          </button>
          <button class="btn-outline" onclick="referCaseToSpecialist()">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="4" width="18" height="18" rx="2" ry="2"/><line x1="16" y1="2" x2="16" y2="6"/><line x1="8" y1="2" x2="8" y2="6"/><line x1="3" y1="10" x2="21" y2="10"/></svg>
            <span>Refer to GI Pathology</span>
          </button>
        </div>
      </div>

      <div class="grid-4">
        <div style="background: var(--color-cream); border-radius: 16px; padding: 18px;">
          <div style="font-size: 12px; color: var(--color-stone);">Tumour Probability</div>
          <div class="serif-heading" style="font-size: 28px; margin-top: 4px;">
            ${r.tumor_probability !== null ? (r.tumor_probability * 100).toFixed(2) + "%" : "—"}
          </div>
          <div style="font-size: 12px; color: var(--color-stone); margin-top: 2px;">Adenocarcinoma (` + "TUM" + `)</div>
        </div>
        <div style="background: var(--color-cream); border-radius: 16px; padding: 18px;">
          <div style="font-size: 12px; color: var(--color-stone);">Top Predicted Tissue</div>
          <div class="serif-heading" style="font-size: 22px; margin-top: 4px; line-height: 1.15;">
            ${r.predicted_class || "Not assessed"}
          </div>
          <div style="font-size: 12px; color: var(--color-stone); margin-top: 2px;">
            ${r.predicted_probability !== null ? (r.predicted_probability * 100).toFixed(2) + "% calibrated" : "Out of scope"}
          </div>
        </div>
        <div style="background: var(--color-cream); border-radius: 16px; padding: 18px;">
          <div style="font-size: 12px; color: var(--color-stone);">H&amp;E Stain Check</div>
          <div class="serif-heading" style="font-size: 24px; margin-top: 4px;">
            ${(r.input_check.he_colour_fraction * 100).toFixed(0)}% H&amp;E Hue
          </div>
          <div style="font-size: 12px; color: var(--color-stone); margin-top: 2px;">${r.input_check.image_type}</div>
        </div>
        <div style="background: var(--color-cream); border-radius: 16px; padding: 18px;">
          <div style="font-size: 12px; color: var(--color-stone);">Stained Tissue Area</div>
          <div class="serif-heading" style="font-size: 28px; margin-top: 4px;">
            ${(r.input_check.tissue_fraction * 100).toFixed(1)}%
          </div>
          <div style="font-size: 12px; color: var(--color-stone); margin-top: 2px;">Chroma: ${r.input_check.colourfulness}</div>
        </div>
      </div>
    </div>

    <div class="grid-2" style="grid-template-columns: 1.35fr 1fr; align-items: start;">
      <div class="white-card" style="display: flex; flex-direction: column; gap: 16px;">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
          <h3 style="font-size: var(--text-heading-sm);">ViT Token Grad-CAM Attribution</h3>
          <div style="display: flex; gap: 6px;">
            <button class="nav-pill-btn ${state.crcViewMode === "overlay" ? "active" : ""}" style="padding: 6px 14px; min-height: 34px; font-size: 12px;" onclick="setCrcViewMode('overlay')">Grad-CAM Overlay</button>
            <button class="nav-pill-btn ${state.crcViewMode === "heatmap_only" ? "active" : ""}" style="padding: 6px 14px; min-height: 34px; font-size: 12px;" onclick="setCrcViewMode('heatmap_only')">Raw Attention Field</button>
          </div>
        </div>

        <div class="image-viewer-stage">
          <div class="image-comparison-pair">
            <div class="image-frame-box">
              <div style="font-size: 12px; font-weight: 500; color: var(--color-stone);">Original H&amp;E Patch</div>
              <div class="image-aspect-wrapper">
                <img src="data:image/png;base64,${r.preview_images.original}" alt="Original H&E Patch" />
              </div>
            </div>
            <div class="image-frame-box">
              <div style="font-size: 12px; font-weight: 500; color: var(--color-primary-indigo);">
                ${r.heatmap_target ? `Grad-CAM: ${r.heatmap_target}` : "No Heatmap"}
              </div>
              <div class="image-aspect-wrapper">
                <img src="data:image/png;base64,${activeRightImg}" alt="Grad-CAM Overlay" />
              </div>
            </div>
          </div>
        </div>

        <div style="background: var(--color-cream); border-radius: 12px; padding: 14px 18px; font-size: 13px;">
          <strong>Confidence &amp; Quality Audit (${r.confidence}):</strong>
          <ul style="margin-top: 6px; padding-left: 18px;">${reasonsHtml}</ul>
        </div>
      </div>

      <div class="white-card" style="display: flex; flex-direction: column; gap: 16px;">
        <div>
          <h3 style="font-size: var(--text-heading-sm); margin-bottom: 4px;">Calibrated 9-Class Tissue Scores</h3>
          <p style="font-size: 12px; color: var(--color-stone); margin-bottom: 14px;">Temperature-calibrated linear head over frozen ViT patch tokens</p>
          <div class="prob-bar-list">${probsHtml}</div>
        </div>
      </div>
    </div>

    ${tilesCardHtml}
  `;
}

function recordAnalyzedCase(caseData) {
  const id = `${caseData.modality}-${caseData.filename}-${Date.now()}`;
  const entry = { id, data: caseData };
  state.analyzedCases.unshift(entry);
  if (state.analyzedCases.length > 15) state.analyzedCases.pop();
  updateChatCaseDropdown();
}

function updateChatCaseDropdown() {
  const select = document.getElementById("chat-case-select");
  if (!select) return;
  const options = [
    `<option value="latest">Latest Analyzed Case (${state.analyzedCases[0] ? `${state.analyzedCases[0].data.filename} · ${state.analyzedCases[0].data.finding}` : "None yet"})</option>`,
    `<option value="none">No Case Attached (General Medical Q&amp;A)</option>`,
  ];
  state.analyzedCases.forEach((c, idx) => {
    options.push(
      `<option value="${idx}">#${idx + 1}: [${c.data.modality === "brain_mri" ? "Brain MRI" : "Colorectal H&E"}] ${c.data.filename} — ${c.data.finding}</option>`
    );
  });
  select.innerHTML = options.join("");
}

function getAttachedCaseForChat() {
  const sel = document.getElementById("chat-case-select")?.value || "latest";
  if (sel === "none") return null;
  if (sel === "latest") return state.analyzedCases[0]?.data || null;
  const idx = parseInt(sel, 10);
  return state.analyzedCases[idx]?.data || null;
}

function discussCaseInAssistant() {
  switchTab("assistant");
  const select = document.getElementById("chat-case-select");
  if (select) select.value = "latest";
  sendQuickPrompt("Please explain the attached MedVisionAI analysis, including model consensus and clinical triage priority.");
}

function renderChatMessages() {
  const box = document.getElementById("chat-messages-box");
  if (!box) return;
  box.innerHTML = state.chatHistory
    .map((m) => {
      const htmlContent = formatMarkdownSimple(m.content);
      return `
        <div class="chat-bubble ${m.role}">
          ${htmlContent}
          ${m.provider ? `<div style="margin-top: 10px; font-size: 11px; opacity: 0.72;">Source: ${m.provider}</div>` : ""}
        </div>
      `;
    })
    .join("");
  box.scrollTop = box.scrollHeight;
}

function formatMarkdownSimple(md) {
  if (!md) return "";
  return md
    .replace(/^### (.*$)/gim, "<h3>$1</h3>")
    .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
    .replace(/\*(.*?)\*/g, "<em>$1</em>")
    .replace(/`([^`]+)`/g, "<code style='background:rgba(8,3,49,0.07);padding:2px 6px;border-radius:6px;font-size:13px;'>$1</code>")
    .replace(/^> (.*$)/gim, "<blockquote>$1</blockquote>")
    .replace(/^- (.*$)/gim, "<li>$1</li>")
    .replace(/(<li>.*<\/li>)/gis, "<ul>$1</ul>")
    .replace(/\n\n/g, "<br/><br/>");
}

async function handleChatSubmit(event) {
  event.preventDefault();
  const input = document.getElementById("chat-input-box");
  if (!input || !input.value.trim()) return;
  const text = input.value.trim();
  input.value = "";
  await sendChatMessage(text);
}

async function sendQuickPrompt(text) {
  await sendChatMessage(text);
}

async function sendChatMessage(text) {
  const attached = getAttachedCaseForChat();
  state.chatHistory.push({
    role: "user",
    content: text + (attached ? `\n\n*(Attached Case: ${attached.filename} — ${attached.finding})*` : ""),
  });
  renderChatMessages();

  try {
    const resp = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: text,
        attached_case: attached,
      }),
    });
    const data = await resp.json();
    state.chatHistory.push({
      role: "assistant",
      content: data.reply || "Unable to generate response.",
      provider: data.provider || "MedVisionAI Clinical Engine",
    });
    renderChatMessages();
  } catch (err) {
    state.chatHistory.push({
      role: "assistant",
      content: `Connection error: ${err.message}`,
      provider: "System",
    });
    renderChatMessages();
  }
}

function switchApptSubtab(subtab) {
  const bookView = document.getElementById("appt-book-view");
  const triageView = document.getElementById("appt-triage-view");
  const bookBtn = document.getElementById("subtab-book-btn");
  const triageBtn = document.getElementById("subtab-triage-btn");

  if (subtab === "book") {
    bookView.style.display = "block";
    triageView.style.display = "none";
    bookBtn.className = "btn-primary";
    triageBtn.className = "btn-outline";
  } else {
    bookView.style.display = "none";
    triageView.style.display = "block";
    bookBtn.className = "btn-outline";
    triageBtn.className = "btn-primary";
    renderAppointmentsQueue();
  }
}

function renderSpecialists() {
  const container = document.getElementById("specialists-list-container");
  if (!container) return;
  container.innerHTML = state.specialists
    .map((s) => {
      const isSelected = s.id === state.selectedSpecialistId;
      const pillClass = s.modality === "brain_mri" ? "forest" : "petal";
      return `
        <div class="white-card" style="padding: 22px; border: 2px solid ${isSelected ? "var(--color-primary-indigo)" : "transparent"}; cursor: pointer; transition: border-color 0.2s ease;" onclick="selectSpecialist('${s.id}')">
          <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
            <div style="display: flex; align-items: center; gap: 12px;">
              <div class="brand-mark" style="width: 38px; height: 38px; font-weight: 500; font-size: 13px;">${s.badge}</div>
              <div>
                <h4 style="font-size: 19px; line-height: 1.2;">${s.title}</h4>
                <span style="font-size: 12px; color: var(--color-stone);">${s.department} · ${s.hospital}</span>
              </div>
            </div>
            <span class="info-pill ${pillClass}" style="padding: 4px 10px; font-size: 10px;">
              ${s.modality === "brain_mri" ? "NEURO MRI" : "GI PATHOLOGY"}
            </span>
          </div>
          <p style="font-size: 13px; color: var(--color-stone);">${s.focus}</p>
        </div>
      `;
    })
    .join("");
}

function selectSpecialist(id) {
  state.selectedSpecialistId = id;
  renderSpecialists();
  const spec = state.specialists.find((s) => s.id === id);
  const banner = document.getElementById("selected-specialist-banner");
  if (spec && banner) {
    banner.textContent = `Selected: ${spec.title} · ${spec.department} (${spec.hospital})`;
  }
}

function fillLatestCaseIntoAppointment() {
  const latest = state.analyzedCases[0]?.data;
  if (!latest) return;
  const summaryInput = document.getElementById("appt-summary-input");
  const urgencySelect = document.getElementById("appt-urgency-select");
  if (latest.modality === "brain_mri") {
    const cls = latest.classification.predicted_class;
    const prob = (latest.classification.probability * 100).toFixed(1);
    const area = latest.segmentation.tumor_area_percent.toFixed(2);
    const rscore = latest.risk_summary.risk_score;
    summaryInput.value = `Brain MRI (${latest.filename}): Predicted ${cls} (${prob}%), U-Net area ${area}%, Risk Score ${rscore}/100 (${latest.risk_summary.risk_level}).`;
    urgencySelect.value = cls === "glioma" || latest.risk_summary.risk_level === "high" ? "urgent" : cls === "no_tumor" ? "routine" : "soon";
    selectSpecialist("neuro-oncology");
  } else {
    const prob = latest.predicted_probability ? (latest.predicted_probability * 100).toFixed(1) + "%" : "N/A";
    summaryInput.value = `Colorectal H&E (${latest.filename}): ${latest.finding}; Predicted ${latest.predicted_class || "N/A"} (${prob}).`;
    urgencySelect.value = latest.finding === "Tumour tissue detected" ? "urgent" : "routine";
    selectSpecialist("gi-pathology");
  }
}

function referCaseToSpecialist() {
  switchTab("appointments");
  switchApptSubtab("book");
  fillLatestCaseIntoAppointment();
}

async function submitAppointmentRequest(event) {
  event.preventDefault();
  const payload = {
    specialist_id: state.selectedSpecialistId,
    urgency: document.getElementById("appt-urgency-select").value,
    requester_name: document.getElementById("appt-requester-input").value,
    contact: document.getElementById("appt-contact-input").value,
    patient_ref: document.getElementById("appt-ref-input").value,
    preferred: document.getElementById("appt-pref-input").value,
    case_summary: document.getElementById("appt-summary-input").value,
    reason: document.getElementById("appt-reason-input").value,
  };

  try {
    const resp = await fetch("/api/appointments", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!resp.ok) throw new Error("Failed to submit appointment request");
    const data = await resp.json();
    state.appointments = data.appointments || [];
    document.getElementById("appt-reason-input").value = "";
    renderAppointmentsQueue();
    switchApptSubtab("triage");
  } catch (err) {
    alert(err.message);
  }
}

function renderAppointmentsQueue() {
  const badge = document.getElementById("nav-triage-count");
  const pending = state.appointments.filter((a) => a.status === "pending");
  if (badge) badge.textContent = pending.length;

  const kpiRow = document.getElementById("triage-kpi-row");
  if (kpiRow) {
    const urgentCount = pending.filter((a) => a.urgency === "urgent").length;
    const soonCount = pending.filter((a) => a.urgency === "soon").length;
    const routineCount = pending.filter((a) => a.urgency === "routine").length;
    const confirmedCount = state.appointments.filter((a) => a.status === "confirmed").length;

    kpiRow.innerHTML = `
      <div class="white-card" style="padding: 20px; text-align: center;">
        <div style="font-size: 12px; color: var(--color-stone);">Urgent Waiting (≤48h)</div>
        <div class="serif-heading" style="font-size: 36px; color: var(--color-ember);">${urgentCount}</div>
      </div>
      <div class="white-card" style="padding: 20px; text-align: center;">
        <div style="font-size: 12px; color: var(--color-stone);">Soon Waiting (≤2w)</div>
        <div class="serif-heading" style="font-size: 36px;">${soonCount}</div>
      </div>
      <div class="white-card" style="padding: 20px; text-align: center;">
        <div style="font-size: 12px; color: var(--color-stone);">Routine Waiting (≤6w)</div>
        <div class="serif-heading" style="font-size: 36px;">${routineCount}</div>
      </div>
      <div class="white-card" style="padding: 20px; text-align: center;">
        <div style="font-size: 12px; color: var(--color-stone);">Confirmed Consults</div>
        <div class="serif-heading" style="font-size: 36px; color: var(--color-forest);">${confirmedCount}</div>
      </div>
    `;
  }

  const qContainer = document.getElementById("appointments-queue-container");
  if (!qContainer) return;

  qContainer.innerHTML = state.appointments
    .map((a) => {
      const urgClass = a.urgency === "urgent" ? "ember" : a.urgency === "soon" ? "petal" : "blue";
      const statusClass = a.status === "confirmed" ? "forest" : a.status === "pending" ? "blue" : "";
      return `
        <div class="white-card" style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 20px;">
          <div style="flex: 1; min-width: 300px;">
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 10px; flex-wrap: wrap;">
              ${a.queue_position ? `<span class="info-pill" style="background: var(--color-deep-ink); color: #fff;">QUEUE #${a.queue_position}</span>` : ""}
              <span class="info-pill ${urgClass}">${a.urgency.toUpperCase()} · ${a.urgency_info?.target || ""}</span>
              <span class="info-pill ${statusClass}">STATUS: ${a.status.toUpperCase()}</span>
              ${a.patient_ref ? `<span class="info-pill">REF: ${a.patient_ref}</span>` : ""}
            </div>
            <h3 style="font-size: 22px; margin-bottom: 4px;">${a.specialist} · ${a.department}</h3>
            <p style="font-size: 13px; color: var(--color-stone); margin-bottom: 12px;">
              Requested by <strong>${a.requester_name}</strong> (${a.contact}) · ${a.hospital}
            </p>
            <p style="font-size: 14px; margin-bottom: 10px;"><strong>Clinical Reason:</strong> ${a.reason}</p>
            ${a.case_summary ? `<div style="background: var(--color-cream); padding: 10px 14px; border-radius: 10px; font-size: 12px; margin-bottom: 8px;"><strong>Attached AI Summary:</strong> ${a.case_summary}</div>` : ""}
            ${a.appointment_at ? `<div style="font-size: 13px; color: var(--color-forest); font-weight: 500;">Confirmed Slot: ${a.appointment_at} · Location: ${a.location}</div>` : ""}
          </div>

          <div style="display: flex; flex-direction: column; gap: 8px; min-width: 200px;">
            ${
              a.status === "pending"
                ? `
              <button class="btn-primary" style="padding: 10px 16px; font-size: 12px;" onclick="updateApptStatus('${a.id}', 'confirmed')">Confirm Slot (Tomorrow 10:00)</button>
              <button class="btn-outline" style="padding: 8px 16px; font-size: 12px;" onclick="updateApptStatus('${a.id}', 'declined')">Decline Request</button>
            `
                : a.status === "confirmed"
                ? `
              <button class="btn-outline" style="padding: 8px 16px; font-size: 12px;" onclick="updateApptStatus('${a.id}', 'completed')">Mark Completed</button>
            `
                : ""
            }
          </div>
        </div>
      `;
    })
    .join("");
}

async function updateApptStatus(id, status) {
  const body = { status };
  if (status === "confirmed") {
    body.appointment_at = "2026-10-26 10:00";
    body.location = "Multidisciplinary Oncology Consult Suite 3";
    body.staff_note = "Confirmed via MedVisionAI Coordinator Triage.";
  }
  const resp = await fetch(`/api/appointments/${id}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (resp.ok) {
    const data = await resp.json();
    state.appointments = data.appointments || [];
    renderAppointmentsQueue();
  }
}

function renderBenchmarks() {
  const container = document.getElementById("benchmarks-content-container");
  if (!container || !state.benchmarks) return;

  const bt = state.benchmarks.brain_tumor;
  const crc = state.benchmarks.colorectal;

  const cnnBackbonesRows = (bt.cnn_ensemble_summary.backbones || [])
    .map(
      (b) => `
      <tr>
        <td><strong>${b.backbone_name}</strong> (${b.run_name})</td>
        <td>Epoch ${b.best_epoch}</td>
        <td>${(b.val_accuracy * 100).toFixed(2)}%</td>
        <td>${(b.val_macro_f1 * 100).toFixed(2)}%</td>
        <td><strong>${(b.test_accuracy_tta * 100).toFixed(2)}%</strong></td>
        <td><strong>${(b.test_macro_f1_tta * 100).toFixed(2)}%</strong></td>
      </tr>
    `
    )
    .join("");

  const classicalRows = (bt.classical_ml || [])
    .map(
      (m) => `
      <tr>
        <td><strong>${m.model}</strong></td>
        <td>${(m.accuracy * 100).toFixed(2)}%</td>
        <td>${(m.precision * 100).toFixed(2)}%</td>
        <td>${(m.recall * 100).toFixed(2)}%</td>
        <td><strong>${(m.f1_score * 100).toFixed(2)}%</strong></td>
        <td>${(m.auc_roc * 100).toFixed(2)}%</td>
        <td>${m.training_time_s}s</td>
      </tr>
    `
    )
    .join("");

  const crcRobustnessRows = Object.entries(crc.phikon?.robustness || {})
    .map(([cond, pMetrics]) => {
      const lMetrics = crc.lunit_vits8?.robustness?.[cond] || {};
      return `
        <tr>
          <td><strong>${cond}</strong></td>
          <td>${(pMetrics.accuracy_9class * 100).toFixed(1)}% · ${(pMetrics.tumor_recall * 100).toFixed(0)}% recall</td>
          <td>${((lMetrics.accuracy_9class || 0) * 100).toFixed(1)}% · ${((lMetrics.tumor_recall || 0) * 100).toFixed(0)}% recall</td>
        </tr>
      `;
    })
    .join("");

  const graphsGridHtml = (bt.graphs || [])
    .map(
      (g) => `
      <div class="white-card" style="padding: 18px; display: flex; flex-direction: column; gap: 10px;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
          <span class="info-pill blue" style="padding: 3px 10px; font-size: 10px;">${g.category}</span>
        </div>
        <h4 style="font-size: 18px;">${g.title}</h4>
        <div style="background: var(--color-cream); border-radius: 12px; padding: 8px; overflow: hidden;">
          <img src="${g.url}" alt="${g.title}" style="width: 100%; height: 220px; object-fit: contain; display: block;" loading="lazy" />
        </div>
      </div>
    `
    )
    .join("");

  container.innerHTML = `
    <!-- Top Gradient Hero Summary -->
    <div class="gradient-accent-banner">
      <div class="grid-4" style="text-align: center;">
        <div>
          <div class="serif-heading" style="font-size: 42px; color: #fff;">10,793</div>
          <div style="font-size: 13px; opacity: 0.9;">Processed Brain MRI Slices</div>
        </div>
        <div>
          <div class="serif-heading" style="font-size: 42px; color: #fff;">99.30%</div>
          <div style="font-size: 13px; opacity: 0.9;">5-Fold CNN Ensemble Test Accuracy</div>
        </div>
        <div>
          <div class="serif-heading" style="font-size: 42px; color: #fff;">95.11%</div>
          <div style="font-size: 13px; opacity: 0.9;">Colorectal 9-Class ViT Accuracy</div>
        </div>
        <div>
          <div class="serif-heading" style="font-size: 42px; color: #fff;">100%</div>
          <div style="font-size: 13px; opacity: 0.9;">Colorectal Tumour Recall (CRC-VAL-7K)</div>
        </div>
      </div>
    </div>

    <!-- Brain Tumor Deep Learning & Classical ML Benchmarks -->
    <div class="grid-2">
      <div class="white-card">
        <span class="info-pill forest" style="margin-bottom: 10px;">BRISC 2025 · DEEP LEARNING</span>
        <h3 style="font-size: var(--text-heading-sm); margin-bottom: 12px;">Brain Tumor CNN Backbones &amp; 5-Fold Ensemble</h3>
        <p style="font-size: 13px; color: var(--color-stone); margin-bottom: 16px;">
          Our deployed <strong>5-Fold EfficientNet-B0 Ensemble</strong> trained with Weighted Focal Loss achieves <strong>99.30% test accuracy</strong> and <strong>99.36% macro F1</strong> across Glioma, Meningioma, Pituitary, and No-Tumor classes.
        </p>
        <div class="clinical-table-wrapper">
          <table class="clinical-table">
            <thead>
              <tr>
                <th>Backbone</th>
                <th>Best Epoch</th>
                <th>Val Acc</th>
                <th>Val F1</th>
                <th>Test Acc (TTA)</th>
                <th>Test F1 (TTA)</th>
              </tr>
            </thead>
            <tbody>${cnnBackbonesRows}</tbody>
          </table>
        </div>
      </div>

      <div class="white-card">
        <span class="info-pill petal" style="margin-bottom: 10px;">NCT-CRC-HE-100K · PATHOLOGY FOUNDATION MODELS</span>
        <h3 style="font-size: var(--text-heading-sm); margin-bottom: 12px;">Colorectal ViT Robustness (50 External Patients)</h3>
        <p style="font-size: 13px; color: var(--color-stone); margin-bottom: 16px;">
          Evaluated on 900 external test patches across 6 stain-shift and compression conditions:
        </p>
        <div class="clinical-table-wrapper">
          <table class="clinical-table">
            <thead>
              <tr>
                <th>Perturbation Condition</th>
                <th>Phikon ViT-B/16 + Head</th>
                <th>Lunit DINO ViT-S/8 + Head</th>
              </tr>
            </thead>
            <tbody>${crcRobustnessRows}</tbody>
          </table>
        </div>
      </div>
    </div>

    <!-- Classical ML Table -->
    <div class="white-card">
      <span class="info-pill blue" style="margin-bottom: 10px;">RADIOMIC TABULAR BENCHMARK · 7 ENGINEERED FEATURES</span>
      <h3 style="font-size: var(--text-heading-sm); margin-bottom: 12px;">Brain MRI Classical Machine Learning Models</h3>
      <div class="clinical-table-wrapper">
        <table class="clinical-table">
          <thead>
            <tr>
              <th>Model</th>
              <th>Test Accuracy</th>
              <th>Precision</th>
              <th>Recall</th>
              <th>F1 Score</th>
              <th>ROC-AUC</th>
              <th>Train Time</th>
            </tr>
          </thead>
          <tbody>${classicalRows}</tbody>
        </table>
      </div>
    </div>

    <!-- 18 Research Figures & EDA Gallery -->
    <div>
      <div class="section-header-centered" style="margin-bottom: 24px;">
        <span class="info-pill">EXPLORATORY DATA ANALYSIS &amp; MODEL DIAGNOSTICS</span>
        <h2 class="section-headline" style="font-size: 36px;">BRISC 2025 Research Figures Gallery</h2>
      </div>
      <div class="grid-3">
        ${graphsGridHtml}
      </div>
    </div>
  `;
}

document.addEventListener("DOMContentLoaded", initApp);
