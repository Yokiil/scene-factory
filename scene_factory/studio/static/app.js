/**
 * SceneFactory v2 Precision 3D Studio Controller
 */

import { StudioViewport } from "/viewer.js";

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => [...document.querySelectorAll(sel)];

const state = {
  xml: "",
  title: "",
  analysis: null,
  verification: null,
  trace: null,
  variants: [],
  cordis: [],
  refImages: [],
  selectedBody: null,
  activeVariantIdx: -1,
};

let toastTimer = null;
function showToast(msg) {
  const el = $("#toast");
  el.textContent = msg;
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.hidden = true; }, 2800);
}

async function api(url, options = {}) {
  const res = await fetch(url, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  const data = await res.json();
  if (!res.ok || data.ok === false) {
    throw new Error(data.error || `HTTP ${res.status}`);
  }
  return data;
}

// Initialize 3D Viewport
const viewport = new StudioViewport($("#three-canvas"), {
  onSelect: (name) => selectBody(name),
});

function setStageMode(mode) {
  $$(".stage-switcher button").forEach((b) =>
    b.setAttribute("aria-selected", String(b.dataset.stage === mode)),
  );
  $$(".stage-view").forEach((v) => v.classList.toggle("active", v.id === `view-${mode}`));
  if (mode === "3d") viewport.resize();
}

function setInspectorTab(tab) {
  $$(".inspector-tabs button").forEach((b) =>
    b.setAttribute("aria-selected", String(b.dataset.tab === tab)),
  );
  $$(".inspector-tab").forEach((p) => p.classList.toggle("active", p.id === `tab-${tab}`));
}

function applyScenePayload(payload, { preserveVariants = false, preserveTrace = false } = {}) {
  state.xml = payload.xml || "";
  state.title = payload.title || "MuJoCo Scene";
  state.analysis = payload.analysis || null;
  state.verification = payload.verification || null;
  if (!preserveTrace && payload.trace !== undefined) {
    state.trace = payload.trace;
  }
  if (!preserveVariants && payload.variants !== undefined) {
    state.variants = payload.variants || [];
    state.activeVariantIdx = -1;
  }

  // Auto-enable nested SE(3) triad axes when inspecting hierarchical depth > 0 scenes
  const maxDepth = Object.values((state.analysis && state.analysis.bodies) || {}).reduce(
    (m, b) => Math.max(m, b.depth || 0),
    0,
  );
  const axesBtn = $('[data-layer="axes"]');
  if (axesBtn) {
    const showAxes = maxDepth > 0;
    axesBtn.classList.toggle("active", showAxes);
    viewport.setLayer("axes", showAxes);
  }

  // 1. Update 3D Viewport
  const stats = viewport.loadScene(payload.scene_3d || { bodies: [], meshes: {} }, state.analysis);
  $("#hud-scene-title").textContent = state.title;
  $("#hud-geom-stats").textContent = `${stats.bodies} bodies · ${stats.triangles.toLocaleString()} tris`;
  $("#hud-selected").hidden = true;

  // 2. Update XML Editor
  $("#xml-editor").value = state.xml;
  $("#xml-meta").textContent = `scene.xml · ${(state.xml.length / 1024).toFixed(1)} KB · sha256:${(payload.xml_sha256 || "").slice(0, 10)}`;

  // 3. Update Telemetry Pill & KPIs
  renderTelemetry();
  renderSpatialTab();
  renderMuJoCoRenders();
  renderTraceTab();
  renderVariantsTab();
}

function renderTelemetry() {
  const a = state.analysis;
  const v = state.verification;
  const pill = $("#telemetry-pill");
  const cols = a ? a.colliding_pairs.length : 0;
  const flts = a ? a.floating_bodies.length : 0;
  const compiled = v ? v.compiled : true;
  const score = v ? Number(v.score).toFixed(2) : "—";

  $("#tel-compile").textContent = compiled ? "Compiled" : "Compile Error";
  $("#tel-collision").textContent = cols > 0 ? `${cols} Collision${cols > 1 ? "s" : ""}` : flts > 0 ? `${flts} Floating` : "0 Collisions";
  $("#tel-score").textContent = `Score ${score}`;

  if (!compiled || cols > 0) pill.dataset.state = "danger";
  else if (flts > 0 || (v && !v.settled_stable)) pill.dataset.state = "warn";
  else pill.dataset.state = "ok";
}

function renderSpatialTab() {
  const a = state.analysis;
  if (!a) return;
  const bodyCount = Object.keys(a.bodies || {}).length;
  const cols = a.colliding_pairs || [];
  const flts = a.floating_bodies || [];

  const kBodies = $("#kpi-bodies");
  const kCols = $("#kpi-collisions");
  const kFlts = $("#kpi-floating");
  kBodies.textContent = String(bodyCount);
  kCols.textContent = String(cols.length);
  kFlts.textContent = String(flts.length);
  kCols.className = `mono ${cols.length ? "danger" : "ok"}`;
  kFlts.className = `mono ${flts.length ? "warn" : "ok"}`;

  const badge = $("#diag-badge");
  if (cols.length) {
    badge.className = "micro-pill danger";
    badge.textContent = `${cols.length} Colliding`;
  } else if (flts.length) {
    badge.className = "micro-pill warn";
    badge.textContent = `${flts.length} Floating`;
  } else {
    badge.className = "micro-pill ok";
    badge.textContent = "Assembly Valid";
  }

  // Diagnostics cards
  const diagEl = $("#diagnostics-list");
  diagEl.innerHTML = "";
  if (!cols.length && !flts.length) {
    diagEl.innerHTML = `<div class="diag-card ok"><strong>零穿模 · 支撑稳定</strong><span>Analytical SE(3)+OBB 与 MuJoCo 交叉验证通过</span></div>`;
  } else {
    for (const pair of cols) {
      const card = document.createElement("div");
      card.className = "diag-card danger";
      const mj = pair.mujoco_signed_distance_m != null ? ` · MuJoCo=${(pair.mujoco_signed_distance_m * 1000).toFixed(1)}mm` : "";
      card.innerHTML = `<strong>穿模：${pair.body_a} ↔ ${pair.body_b}</strong><span>depth ${(pair.penetration_m * 1000).toFixed(1)} mm${mj} · axis [${pair.separation_axis.map((x) => x.toFixed(2)).join(", ")}]</span>`;
      card.addEventListener("click", () => {
        viewport.focusBody(pair.body_a);
        selectBody(pair.body_a);
      });
      diagEl.appendChild(card);
    }
    for (const flt of flts) {
      const card = document.createElement("div");
      card.className = "diag-card warn";
      card.innerHTML = `<strong>悬空：${flt.body}</strong><span>clearance ${(flt.clearance_m * 1000).toFixed(1)} mm above ${flt.supporter}</span>`;
      card.addEventListener("click", () => {
        viewport.focusBody(flt.body);
        selectBody(flt.body);
      });
      diagEl.appendChild(card);
    }
  }

  // Nested SE(3) Body Hierarchy Tree
  const treeEl = $("#body-tree");
  treeEl.innerHTML = "";
  const collidedSet = new Set(cols.flatMap((p) => [p.body_a, p.body_b]));
  const floatingSet = new Set(flts.map((f) => f.body));
  const bodies = Object.values(a.bodies || {});
  const maxDepth = bodies.reduce((m, b) => Math.max(m, b.depth || 0), 0);
  $("#depth-badge").textContent = maxDepth > 0 ? `Nested Depth ${maxDepth}` : "Flat Depth 0";

  // Preserve parent-child DFS ordering
  const byParent = new Map();
  for (const b of bodies) {
    const key = b.parent || "__root__";
    if (!byParent.has(key)) byParent.set(key, []);
    byParent.get(key).push(b);
  }
  const ordered = [];
  const visit = (parentKey) => {
    for (const b of byParent.get(parentKey) || []) {
      ordered.push(b);
      visit(b.name);
    }
  };
  visit("__root__");

  for (const b of ordered) {
    const row = document.createElement("button");
    row.type = "button";
    row.className = "tree-row";
    row.dataset.body = b.name;
    const indent = b.depth > 0 ? `<span class="tree-indent">${"└─ ".padStart(b.depth * 2 + 1, "  ")}</span>` : "";
    const tags = [
      `<span class="tag-mini ${b.dynamic ? "dyn" : ""}">${b.dynamic ? "DYN" : "FIX"}</span>`,
      collidedSet.has(b.name) ? `<span class="tag-mini bad">COL</span>` : "",
      floatingSet.has(b.name) ? `<span class="tag-mini flt">FLT</span>` : "",
    ].join("");
    row.innerHTML = `<span class="tree-name">${indent}${b.name}</span><span class="tree-tags">${tags}</span>`;
    row.addEventListener("click", () => {
      viewport.focusBody(b.name);
      selectBody(b.name);
    });
    treeEl.appendChild(row);
  }
  $("#selected-detail").hidden = true;
}

function selectBody(name) {
  state.selectedBody = name;
  $$(".tree-row").forEach((r) => r.classList.toggle("selected", r.dataset.body === name));
  const chip = $("#hud-selected");
  const detailSec = $("#selected-detail");
  if (!name || !state.analysis || !state.analysis.bodies[name]) {
    chip.hidden = true;
    detailSec.hidden = true;
    return;
  }
  const b = state.analysis.bodies[name];
  const ss = (state.analysis.camera_sspace_coords || {})[name];
  chip.textContent = `Selected · ${name}`;
  chip.hidden = false;
  detailSec.hidden = false;
  $("#detail-name").textContent = `${name} (depth ${b.depth})`;
  $("#detail-grid").innerHTML = `
    <div><small>WORLD CENTER (m)</small><span>[${b.world_center_m.map((v) => v.toFixed(3)).join(", ")}]</span></div>
    <div><small>LOCAL POS (m)</small><span>[${b.local_pos_m.map((v) => v.toFixed(3)).join(", ")}]</span></div>
    <div><small>HALF EXTENTS (m)</small><span>[${b.aabb_half_extents_m.map((v) => v.toFixed(3)).join(", ")}]</span></div>
    <div><small>S-SPACE (H / V / D)</small><span>${ss ? `${ss.horizontal.toFixed(2)} / ${ss.vertical.toFixed(2)} / ${ss.distance.toFixed(2)}` : "—"}</span></div>
  `;
}

function renderMuJoCoRenders() {
  const v = state.verification;
  const metricsEl = $("#renders-metrics");
  const gridEl = $("#renders-grid");
  if (!v) return;
  metricsEl.innerHTML = `
    <div class="metric-box"><small>SETTLE STEPS</small><strong>${v.settle_steps} (${v.sim_time_s}s)</strong></div>
    <div class="metric-box"><small>MAX PENETRATION</small><strong>${(v.final_max_penetration_m * 1000).toFixed(2)} mm</strong></div>
    <div class="metric-box"><small>MAX DISPLACEMENT</small><strong>${(v.max_displacement_m * 1000).toFixed(2)} mm</strong></div>
    <div class="metric-box"><small>VERIFIER SCORE</small><strong>${Number(v.score).toFixed(2)}</strong></div>
  `;
  gridEl.innerHTML = "";
  const renders = v.renders || [];
  if (!renders.length) {
    gridEl.innerHTML = `<div class="empty-note">${v.render_error || "暂无离屏 MuJoCo 渲染图"}</div>`;
    return;
  }
  for (const item of renders) {
    const fig = document.createElement("figure");
    fig.className = "render-card";
    fig.innerHTML = `
      <img src="${item.url}" alt="${item.camera}" />
      <figcaption><span>${item.camera}</span><span>${item.width}×${item.height} · ${(item.sha256 || "").slice(0, 8)}</span></figcaption>
    `;
    gridEl.appendChild(fig);
  }
}

function renderTraceTab() {
  const t = state.trace;
  const btnExport = $("#btn-export-sft");
  const listEl = $("#trace-timeline");
  if (!t) {
    $("#trace-meta").textContent = "无活跃 Trace";
    btnExport.disabled = true;
    listEl.innerHTML = `<div class="empty-note">当前场景未附带多轮 Trace。点击底部「运行离线自纠 Demo」或接入 VLM 生成即可查看完整可训练轨迹。</div>`;
    return;
  }
  btnExport.disabled = false;
  $("#trace-meta").textContent = `${t.model} · ${t.turns} turns · reward ${Number(t.verifier_score).toFixed(2)}`;
  listEl.innerHTML = "";

  for (const msg of t.messages || []) {
    if (msg.role === "system") continue;
    const step = document.createElement("div");
    step.className = "trace-step";

    let bodyHtml = "";
    let toolsHtml = "";
    let thumbsHtml = "";
    if (typeof msg.content === "string") {
      if (msg.role === "tool") {
        try {
          const parsed = JSON.parse(msg.content);
          if (parsed.verification) {
            const vr = parsed.verification;
            bodyHtml = `MuJoCo verify → compiled=${vr.compiled}, stable=${vr.settled_stable}, max_disp=${(vr.max_displacement_m * 1000).toFixed(1)}mm, pen=${(vr.initial_max_penetration_m * 1000).toFixed(1)}mm, score=${vr.score}`;
          } else if (parsed.inserted) {
            bodyHtml = `insert_asset(${parsed.inserted}) · detailed_mesh=${parsed.detailed_mesh} · cache_hit=${parsed.fragment_cache_hit}`;
          } else if (parsed.assets) {
            bodyHtml = `search_assets → ${parsed.count} matches (${parsed.assets.slice(0, 5).map((a) => a.asset_id).join(", ")})`;
          } else {
            bodyHtml = msg.content.slice(0, 180);
          }
        } catch {
          bodyHtml = msg.content.slice(0, 180);
        }
      } else {
        bodyHtml = msg.content;
      }
    } else if (Array.isArray(msg.content)) {
      const texts = msg.content.filter((p) => p.type === "text").map((p) => p.text).join(" ");
      bodyHtml = texts;
      const imgs = msg.content.filter((p) => p.type === "image_ref" && p.url);
      if (imgs.length) {
        thumbsHtml = `<div class="trace-thumbs">${imgs.map((i) => `<img src="${i.url}" alt="${i.camera || "view"}" />`).join("")}</div>`;
      }
    }

    if (msg.tool_calls && msg.tool_calls.length) {
      toolsHtml = `<div class="trace-tools">${msg.tool_calls.map((tc) => `<span class="tool-chip">${tc.function.name}()</span>`).join("")}</div>`;
    }

    const label = msg.role === "tool" ? `TOOL · ${msg.name}` : msg.role.toUpperCase();
    step.innerHTML = `
      <div class="trace-step-head"><span class="role-${msg.role}">${label}</span></div>
      <div class="trace-body">${bodyHtml || "—"}</div>
      ${toolsHtml}
      ${thumbsHtml}
    `;
    listEl.appendChild(step);
  }
}

function renderVariantsTab() {
  const grid = $("#variants-grid");
  grid.innerHTML = "";
  if (!state.variants.length) {
    grid.innerHTML = `<div class="empty-note">暂无随机变体，点击上方「生成 Random 变体」即可原地采样。</div>`;
    return;
  }
  state.variants.forEach((v, idx) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = `variant-tile ${idx === state.activeVariantIdx ? "active" : ""}`;
    btn.innerHTML = `
      ${v.thumb_url ? `<img src="${v.thumb_url}" alt="seed ${v.seed}" />` : ""}
      <div class="variant-tile-meta">
        <strong>seed ${v.seed}</strong>
        <small>pen ${(v.max_penetration_m * 1000).toFixed(1)}mm · ${v.xml_sha256.slice(0, 7)}</small>
      </div>
    `;
    btn.addEventListener("click", async () => {
      state.activeVariantIdx = idx;
      renderVariantsTab();
      try {
        $("#command-status").textContent = `Loading seed ${v.seed}…`;
        const res = await api("/api/analyze", {
          method: "POST",
          body: JSON.stringify({ xml: v.xml, title: `Variant · seed ${v.seed}`, settle_steps: 80, render: true }),
        });
        applyScenePayload(res, { preserveVariants: true, preserveTrace: true });
        $("#command-status").textContent = `Variant seed ${v.seed} loaded`;
      } catch (err) {
        showToast(err.message);
      }
    });
    grid.appendChild(btn);
  });
}

function renderCordisTab() {
  const activeCount = state.cordis.filter((p) => p.enabled).length;
  $("#cordis-count").textContent = String(activeCount);
  const list = $("#cordis-list");
  list.innerHTML = "";
  for (const p of state.cordis) {
    const card = document.createElement("div");
    card.className = "cordis-card";
    card.innerHTML = `
      <div class="cordis-card-top">
        <strong>${p.name}</strong>
        <label class="toggle-switch" aria-label="切换插件 ${p.name}">
          <input type="checkbox" ${p.enabled ? "checked" : ""} />
          <span class="toggle-track"></span>
        </label>
      </div>
      <span class="cordis-kind">${p.kind}</span>
      <p>${p.purpose || ""}</p>
      <div class="cordis-exit">Model: ${p.added_for_model || "general"}<br/>Exit: ${p.exit_criteria || "—"}</div>
    `;
    const input = card.querySelector("input");
    input.addEventListener("change", async () => {
      try {
        const res = await api("/api/cordis/toggle", {
          method: "POST",
          body: JSON.stringify({ name: p.name, enabled: input.checked }),
        });
        state.cordis = res.cordis_plugins;
        renderCordisTab();
        showToast(`${p.name} 已${input.checked ? "挂载" : "热卸载"}`);
      } catch (err) {
        input.checked = !input.checked;
        showToast(err.message);
      }
    });
    list.appendChild(card);
  }
}

function renderRefImages() {
  const el = $("#ref-images");
  el.innerHTML = "";
  el.hidden = state.refImages.length === 0;
  state.refImages.forEach((url, idx) => {
    const div = document.createElement("div");
    div.className = "ref-thumb";
    div.innerHTML = `<img src="${url}" alt="ref ${idx + 1}" /><button type="button" aria-label="移除">×</button>`;
    div.querySelector("button").addEventListener("click", () => {
      state.refImages.splice(idx, 1);
      renderRefImages();
    });
    el.appendChild(div);
  });
}

// Event Bindings
$$(".stage-switcher button").forEach((b) =>
  b.addEventListener("click", () => setStageMode(b.dataset.stage)),
);
$$(".inspector-tabs button").forEach((b) =>
  b.addEventListener("click", () => setInspectorTab(b.dataset.tab)),
);
$("#btn-cordis-jump").addEventListener("click", () => setInspectorTab("cordis"));

$$("[data-cam]").forEach((b) =>
  b.addEventListener("click", () => {
    $$("[data-cam]").forEach((x) => x.classList.remove("active"));
    if (b.dataset.cam !== "reset") b.classList.add("active");
    else $('[data-cam="persp"]').classList.add("active");
    viewport.setCameraPreset(b.dataset.cam);
  }),
);

$$("[data-layer]").forEach((b) =>
  b.addEventListener("click", () => {
    const next = !b.classList.contains("active");
    b.classList.toggle("active", next);
    viewport.setLayer(b.dataset.layer, next);
  }),
);

$("#btn-theme").addEventListener("click", () => {
  const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  viewport.setTheme(next);
});

$("#preset-select").addEventListener("change", async (e) => {
  await loadPreset(e.target.value);
});

$("#image-input").addEventListener("change", (e) => {
  for (const file of e.target.files || []) {
    const reader = new FileReader();
    reader.onload = () => {
      state.refImages.push(String(reader.result));
      renderRefImages();
    };
    reader.readAsDataURL(file);
  }
  e.target.value = "";
});

async function runGeneration({ demoMode = false } = {}) {
  const prompt = $("#prompt-input").value.trim();
  if (!prompt) return;
  const btn = $("#btn-generate");
  btn.disabled = true;
  $("#command-status").textContent = demoMode ? "Running self-correct demo…" : "Running VLM agent…";
  try {
    const res = await api("/api/generate", {
      method: "POST",
      body: JSON.stringify({
        prompt,
        images: state.refImages,
        randomize_count: Number($("#opt-random").value),
        no_plugins: $("#opt-no-plugins").checked,
        demo_mode: demoMode,
      }),
    });
    applyScenePayload(res);
    setInspectorTab("trace");
    $("#command-status").textContent = `Done · Score ${Number(res.verification.score).toFixed(2)}`;
  } catch (err) {
    showToast(err.message);
    $("#command-status").textContent = "Awaiting LLM config";
  } finally {
    btn.disabled = false;
  }
}

$("#command-island").addEventListener("submit", (e) => {
  e.preventDefault();
  runGeneration({ demoMode: false });
});
$("#btn-demo-trace").addEventListener("click", () => runGeneration({ demoMode: true }));

$("#btn-recheck-xml").addEventListener("click", async () => {
  const xml = $("#xml-editor").value;
  try {
    $("#command-status").textContent = "Verifying XML…";
    const res = await api("/api/analyze", {
      method: "POST",
      body: JSON.stringify({ xml, title: `${state.title} (edited)`, settle_steps: 120, render: true }),
    });
    applyScenePayload(res, { preserveVariants: true, preserveTrace: true });
    showToast("XML 空间碰撞与 MuJoCo 校验完成");
    $("#command-status").textContent = "Verified";
  } catch (err) {
    showToast(err.message);
  }
});

$("#btn-copy-xml").addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(state.xml);
    showToast("已复制 MuJoCo XML");
  } catch {
    showToast("复制失败，请手动选中复制");
  }
});

$("#btn-export-xml").addEventListener("click", () => {
  const blob = new Blob([state.xml], { type: "application/xml" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "scene.xml";
  a.click();
  URL.revokeObjectURL(a.href);
});

$("#btn-export-sft").addEventListener("click", () => {
  if (!state.trace) return;
  const line = JSON.stringify({
    trace_id: state.trace.trace_id,
    model: state.trace.model,
    reward: state.trace.verifier_score,
    cordis_plugin_names: (state.trace.cordis_plugins || []).map((p) => p.name),
    messages: state.trace.messages,
    final_xml_sha256: state.trace.final_xml_sha256,
  });
  const blob = new Blob([line + "\n"], { type: "application/jsonl" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `trace_${(state.trace.trace_id || "run").slice(0, 8)}.jsonl`;
  a.click();
  URL.revokeObjectURL(a.href);
});

$("#btn-run-randomize").addEventListener("click", async () => {
  const btn = $("#btn-run-randomize");
  btn.disabled = true;
  try {
    const res = await api("/api/randomize", {
      method: "POST",
      body: JSON.stringify({
        xml: state.xml,
        count: Number($("#rand-count").value),
        seed: Number($("#rand-seed").value),
        xy_jitter_m: Number($("#rand-xy").value),
        yaw_jitter_deg: Number($("#rand-yaw").value),
      }),
    });
    state.variants = res.variants;
    state.activeVariantIdx = -1;
    $("#rand-stats").textContent = `${res.stats.native_compiles} compile · ${res.stats.patched_checks} patches`;
    renderVariantsTab();
    showToast(`已生成 ${res.variants.length} 组碰撞校验变体`);
  } catch (err) {
    showToast(err.message);
  } finally {
    btn.disabled = false;
  }
});

async function loadPreset(presetId) {
  $("#command-status").textContent = "Loading preset…";
  try {
    const res = await api(`/api/preset/${encodeURIComponent(presetId)}`);
    applyScenePayload(res);
    if (presetId.includes("collision") || presetId.includes("floating")) {
      setInspectorTab("spatial");
    }
    $("#command-status").textContent = "Ready";
  } catch (err) {
    showToast(err.message);
    $("#command-status").textContent = "Error";
  }
}

function autoResizePrompt() {
  const el = $("#prompt-input");
  if (!el) return;
  el.style.height = "auto";
  el.style.height = `${Math.min(el.scrollHeight, 88)}px`;
}

$("#prompt-input").addEventListener("input", autoResizePrompt);

async function boot() {
  autoResizePrompt();
  try {
    const info = await api("/api/status");
    state.cordis = info.cordis_plugins || [];
    renderCordisTab();
    await loadPreset("kitchen_high_fidelity");
  } catch (err) {
    showToast(`初始化失败: ${err.message}`);
  }
}

boot();
