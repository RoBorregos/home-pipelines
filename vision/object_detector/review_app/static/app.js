(() => {

  let sseSource  = null;
  let pollTimer  = null;
  let currentRun = localStorage.getItem("od_current_run") || "";

  // ── Toast ─────────────────────────────────────────────────────────────────
  function toast(msg, type = "info") {
    const el = document.getElementById("toast");
    el.textContent = msg;
    el.className = `visible ${type}`;
    if (type !== "error") setTimeout(() => el.className = "", 3000);
  }

  // ── API ───────────────────────────────────────────────────────────────────
  async function api(method, path, body) {
    const res = await fetch(path, {
      method,
      headers: {
        "Content-Type": "application/json",
        "x-api-key": getApiKey(),
        "X-Run": currentRun,
      },
      ...(body ? { body: JSON.stringify(body) } : {}),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || res.statusText);
    }
    return res.json();
  }

  // ── Log panel ─────────────────────────────────────────────────────────────
  function appendLog(line) {
    const body = document.getElementById("log-body");
    if (body.querySelector("p")) body.innerHTML = "";
    const div = document.createElement("div");
    div.className = "log-line";
    if (/error|exception|failed/i.test(line))   div.classList.add("error");
    else if (/warn/i.test(line))                 div.classList.add("warn");
    else if (/\[done\]|complete/i.test(line))    div.classList.add("info-done");
    div.textContent = line;
    body.appendChild(div);
    body.scrollTop = body.scrollHeight;
  }
  window.clearLog = () => {
    document.getElementById("log-body").innerHTML =
      '<p style="color:var(--muted);font-size:0.72rem">Cleared.</p>';
  };
  window.loadLog = async () => {
    if (!currentRun) return;
    expandLog();
    const data = await fetch(`/logs/file?run=${encodeURIComponent(currentRun)}`).then(r => r.json());
    const body = document.getElementById("log-body");
    body.innerHTML = "";
    if (!data.lines?.length) {
      body.innerHTML = '<p style="color:var(--muted);font-size:0.72rem">No log content.</p>';
      return;
    }
    data.lines.forEach(appendLog);
  };

  function expandLog() {
    const panel = document.getElementById("log-panel");
    const btn   = document.getElementById("log-toggle");
    if (panel) panel.classList.remove("collapsed");
    if (btn) btn.textContent = "▼ Hide";
  }
  window.toggleLog = () => {
    const panel = document.getElementById("log-panel");
    const btn   = document.getElementById("log-toggle");
    if (!panel) return;
    const collapsed = panel.classList.toggle("collapsed");
    if (btn) btn.textContent = collapsed ? "▲ Show" : "▼ Hide";
  };

  function startSSE() {
    if (sseSource) { sseSource.close(); sseSource = null; }
    if (!currentRun) return;
    window.clearLog();
    expandLog();
    sseSource = new EventSource(`/logs/stream?run=${encodeURIComponent(currentRun)}`);
    sseSource.onmessage = e => appendLog(e.data);
    sseSource.onerror = () => { sseSource?.close(); sseSource = null; };
  }

  // ── Runs ──────────────────────────────────────────────────────────────────
  async function loadRuns(activeRun = "") {
    const { runs } = await fetch("/runs").then(r => r.json());
    const sel = document.getElementById("run-select");
    sel.innerHTML = '<option value="">— select run —</option>';
    runs.forEach(r => {
      const opt = document.createElement("option");
      opt.value = r.name;
      opt.textContent = `${r.name}  (${r.classes.length} classes${r.has_model ? " ✓ model" : ""})`;
      if (r.name === activeRun) opt.selected = true;
      sel.appendChild(opt);
    });
  }

  window.activateRun = async (name) => {
    // Switch which run this browser is viewing/driving (state lives per-run on the server).
    currentRun = name || "";
    localStorage.setItem("od_current_run", currentRun);
    if (sseSource) { sseSource.close(); sseSource = null; }
    if (!name) { renderStatus({}); return; }
    try {
      const s = await api("POST", `/runs/${name}/activate`, {});
      renderStatus(s);
      toast(`Run "${name}" active`);
    } catch (e) { toast(e.message, "error"); }
  };

  window.openNewRun = () => {
    document.getElementById("modal").classList.add("open");
    document.getElementById("new-run-name").focus();
  };
  window.closeModal = () => document.getElementById("modal").classList.remove("open");

  window.createRun = async () => {
    const name = document.getElementById("new-run-name").value.trim();
    if (!name) return;
    try {
      await api("POST", "/runs", { name });
      closeModal();
      await loadRuns(name);
      toast(`Run "${name}" created and activated`, "success");
      await activateRun(name);
    } catch (e) { toast(e.message, "error"); }
  };

  // ── Status rendering ──────────────────────────────────────────────────────
  const CARDS = ["upload", "segment", "review", "generate", "train"];

  function badge(id, text, type) {
    const el = document.getElementById(`badge-${id}`);
    if (el) { el.textContent = text; el.className = `pill ${type}`; }
  }
  function numEl(id, type, n) {
    const el = document.getElementById(`num-${id}`);
    if (!el) return;
    el.className = `stage-num ${type}`;
    el.textContent = type === "done" ? "✓" : type === "error" ? "!" : n;
  }
  function card(id, type) {
    const el = document.getElementById(`card-${id}`);
    if (el) el.className = `stage-card ${type}`;
  }

  function renderStatus(s) {
    // Point the Review / Test-model links at the currently selected run
    const q = currentRun ? `?run=${encodeURIComponent(currentRun)}` : "";
    const reviewLink = document.getElementById("link-review");
    if (reviewLink) reviewLink.href = `/review${q}`;
    const inferLink = document.getElementById("btn-infer");
    if (inferLink) inferLink.href = `/infer${q}`;

    const running = s.running || "";
    const statusEl = document.getElementById("hdr-status");
    statusEl.textContent = running || (s.error ? "error" : s.run_name ? "idle" : "no run");
    statusEl.className   = `pill ${running ? "running" : s.error ? "error" : "idle"}`;

    // Upload card — always available
    card("upload", ""); badge("upload", "ready", "idle"); numEl("upload", "", 1);

    // Segment card
    if (running === "segment") {
      card("segment", "active"); badge("segment", "running…", "running"); numEl("segment", "active", 2);
    } else if (s.segment_done) {
      card("segment", "done"); badge("segment", "done", "done"); numEl("segment", "done", 2);
    } else if (s.error && !s.segment_done) {
      card("segment", "error"); badge("segment", "error", "error"); numEl("segment", "error", 2);
    } else {
      card("segment", ""); badge("segment", "idle", "idle"); numEl("segment", "", 2);
    }

    // Class chips (segmented status per class)
    const chips = document.getElementById("class-chips");
    chips.innerHTML = "";
    const imported = s.imported_classes || {};
    Object.entries(s.segmented_classes || {}).forEach(([cls, done]) => {
      const c = document.createElement("span");
      if (imported[cls]) {
        c.className = "chip imported";
        c.textContent = `↓ ${cls} (${imported[cls].identifier})`;
      } else {
        c.className = `chip ${done ? "done" : "new"}`;
        c.textContent = done ? `✓ ${cls}` : cls;
      }
      chips.appendChild(c);
    });
    // Same chips on review card
    document.getElementById("review-class-chips").innerHTML = chips.innerHTML;

    // Review card
    if (s.review_done) {
      card("review", "done"); badge("review", "done", "done"); numEl("review", "done", 3);
    } else if (s.segment_done) {
      card("review", "active"); badge("review", "ready", "running"); numEl("review", "active", 3);
    } else {
      card("review", ""); badge("review", "manual", "idle"); numEl("review", "", 3);
    }

    // Generate card
    if (running === "generate") {
      card("generate", "active"); badge("generate", "running…", "running"); numEl("generate", "active", 4);
    } else if (s.generate_done) {
      card("generate", "done"); badge("generate", "done", "done"); numEl("generate", "done", 4);
    } else {
      card("generate", ""); badge("generate", "idle", "idle"); numEl("generate", "", 4);
    }

    // Train card
    if (running === "train") {
      card("train", "active"); badge("train", "running…", "running"); numEl("train", "active", 5);
      document.getElementById("btn-infer").style.display = "none";
    } else if (s.train_done) {
      card("train", "done"); badge("train", "done", "done"); numEl("train", "done", 5);
      document.getElementById("btn-infer").style.display = "";
      document.getElementById("best-weights").textContent = s.best_weights ? "best.pt ready" : "";
    } else {
      card("train", ""); badge("train", "idle", "idle"); numEl("train", "", 5);
      document.getElementById("btn-infer").style.display = "none";
    }

    // Button states
    const busy = !!running;
    document.getElementById("btn-upload").disabled   = busy || !s.run_name;
    document.getElementById("btn-segment").disabled  = busy || !s.run_name;
    document.getElementById("btn-generate").disabled = busy || !s.review_done;
    document.getElementById("btn-train").disabled    = busy || !s.generate_done;
    document.getElementById("btn-approve").disabled  = !s.segment_done || s.review_done;

    if (s.error) toast(`Error: ${s.error}`, "error");
  }

  // ── Polling ───────────────────────────────────────────────────────────────
  async function poll() {
    if (!currentRun) { renderStatus({}); return; }
    try {
      const s = await fetch(`/status?run=${encodeURIComponent(currentRun)}`).then(r => r.json());
      renderStatus(s);
      return s;
    } catch (_) {}
  }
  function startPolling() {
    if (pollTimer) clearInterval(pollTimer);
    pollTimer = setInterval(poll, 2500);
  }

  // ── Stage actions ─────────────────────────────────────────────────────────
  window.doUpload = async () => {
    const url = document.getElementById("cfg-drive-url").value.trim();
    if (!url) { toast("Paste a Drive URL", "error"); return; }
    try {
      badge("upload", "downloading…", "running");
      document.getElementById("btn-upload").disabled = true;
      await api("POST", "/upload/gdrive", { drive_url: url });
      toast("Download started — check logs");
      startSSE();
    } catch (e) { toast(e.message, "error"); }
  };

  window.runSegment = async () => {
    try {
      await api("POST", "/stage/segment/run", {});
      toast("Segmentation started");
      startSSE();
      await poll();
    } catch (e) { toast(e.message, "error"); }
  };

  window.doApprove = async () => {
    try {
      await api("POST", "/review/approve", {});
      toast("Marked as reviewed ✓", "success");
      await poll();
    } catch (e) { toast(e.message, "error"); }
  };

  window.runGenerate = async () => {
    const n = parseInt(document.getElementById("cfg-n-images").value) || 15000;
    try {
      await api("POST", "/stage/generate/run", { images_to_generate: n });
      toast("Generation started");
      startSSE();
      await poll();
    } catch (e) { toast(e.message, "error"); }
  };

  window.runTrain = async () => {
    try {
      await api("POST", "/stage/train/run", {
        device: document.getElementById("cfg-device").value.trim() || "0",
        epochs: parseInt(document.getElementById("cfg-epochs").value) || 100,
        batch:  parseInt(document.getElementById("cfg-batch").value)  || 64,
      });
      toast("Training started");
      startSSE();
      await poll();
    } catch (e) { toast(e.message, "error"); }
  };

  window.doReset = async () => {
    if (!confirm("Reset running state? Files are not deleted.")) return;
    try { await api("POST", "/pipeline/reset", {}); toast("Reset"); await poll(); }
    catch (e) { toast(e.message, "error"); }
  };

  // ── Repository ────────────────────────────────────────────────────────────

  window.toggleRepoPanel = () => {
    const panel = document.getElementById("repo-panel");
    const btn   = document.getElementById("btn-repo-toggle");
    if (!panel) return;
    const visible = panel.style.display !== "none";
    panel.style.display = visible ? "none" : "";
    btn.textContent = visible ? "▼ Show" : "▲ Hide";
    if (!visible) loadRepo();
  };

  async function loadRepo() {
    try {
      const data = await fetch("/repo").then(r => r.json());
      const entries = data.entries || {};
      const labels = Object.keys(entries).sort();
      const list = document.getElementById("repo-checklist");
      list.innerHTML = "";

      labels.forEach(label => {
        const ids = Object.keys(entries[label].identifiers || {}).sort();
        const row = document.createElement("label");
        row.style.cssText =
          "display:flex;align-items:center;gap:8px;padding:6px 10px;" +
          "border-bottom:1px solid var(--border);font-size:0.82rem;cursor:pointer";

        const chk = document.createElement("input");
        chk.type = "checkbox";
        chk.className = "repo-chk";
        chk.dataset.label = label;
        chk.addEventListener("change", updateRepoSelection);

        const name = document.createElement("span");
        name.textContent = label;
        name.style.flex = "1";

        row.appendChild(chk);
        row.appendChild(name);

        if (ids.length > 1) {
          const sel = document.createElement("select");
          sel.className = "repo-id";
          sel.dataset.label = label;
          sel.style.cssText = "font-size:0.76rem;padding:2px 6px";
          ids.forEach(id => {
            const opt = document.createElement("option");
            opt.value = id;
            opt.textContent = `${id} (${entries[label].identifiers[id].image_count})`;
            sel.appendChild(opt);
          });
          // Don't toggle the checkbox when interacting with the dropdown
          sel.addEventListener("click", e => e.preventDefault());
          row.appendChild(sel);
        } else {
          chk.dataset.identifier = ids[0] || "";
          const idTxt = document.createElement("span");
          idTxt.className = "cfg-label";
          idTxt.textContent = `${ids[0] || "—"} (${ids[0] ? entries[label].identifiers[ids[0]].image_count : 0})`;
          row.appendChild(idTxt);
        }

        list.appendChild(row);
      });

      const hint = document.getElementById("repo-empty-hint");
      if (hint) hint.style.display = labels.length ? "none" : "";
      list.style.display = labels.length ? "" : "none";
      updateRepoSelection();
    } catch (_) {}
  }

  function updateRepoSelection() {
    const n = document.querySelectorAll(".repo-chk:checked").length;
    const countEl = document.getElementById("repo-count");
    if (countEl) countEl.textContent = n ? `${n} selected` : "";
    document.getElementById("btn-repo-import").disabled = n === 0;
  }

  window.repoSelectAll = () => {
    document.querySelectorAll(".repo-chk").forEach(c => c.checked = true);
    updateRepoSelection();
  };
  window.repoClearAll = () => {
    document.querySelectorAll(".repo-chk").forEach(c => c.checked = false);
    updateRepoSelection();
  };

  window.doRepoImport = async () => {
    const imports = [];
    document.querySelectorAll(".repo-chk:checked").forEach(chk => {
      const label = chk.dataset.label;
      const sel = document.querySelector(`select.repo-id[data-label="${label}"]`);
      const identifier = sel ? sel.value : chk.dataset.identifier;
      if (identifier) imports.push({ label, identifier });
    });
    if (!imports.length) return;
    try {
      document.getElementById("btn-repo-import").disabled = true;
      const result = await api("POST", "/repo/import", { imports });
      const names = result.imported.map(e => `${e.label}/${e.identifier} (${e.count})`).join(", ");
      toast(`Imported: ${names}`, "success");
      await poll();
      await loadRepo();
    } catch (e) {
      toast(e.message, "error");
      updateRepoSelection();
    }
  };


  // ── Init ──────────────────────────────────────────────────────────────────
  (async () => {
    await loadRuns(currentRun);
    const sel = document.getElementById("run-select");
    if (currentRun && sel.value !== currentRun) {
      // stored run no longer exists on the server
      currentRun = "";
      localStorage.removeItem("od_current_run");
    }
    const s = await poll();
    startPolling();
    if (s?.running) { startSSE(); }
  })();
})();
