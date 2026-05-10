(() => {
  // API key stored in session so you only type it once per browser session
  const API_KEY = (() => {
    let k = sessionStorage.getItem("api_key");
    if (!k) { k = prompt("API key:") || ""; sessionStorage.setItem("api_key", k); }
    return k;
  })();

  let sseSource  = null;
  let pollTimer  = null;

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
      headers: { "Content-Type": "application/json", "x-api-key": API_KEY },
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
    expandLog();
    const data = await fetch("/logs/file").then(r => r.json());
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
    window.clearLog();
    expandLog();
    sseSource = new EventSource("/logs/stream");
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
    if (!name) return;
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
      await poll();
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
    Object.entries(s.segmented_classes || {}).forEach(([cls, done]) => {
      const c = document.createElement("span");
      c.className = `chip ${done ? "done" : "new"}`;
      c.textContent = done ? `✓ ${cls}` : cls;
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
    } else if (s.train_done) {
      card("train", "done"); badge("train", "done", "done"); numEl("train", "done", 5);
      if (s.best_weights) {
        document.getElementById("best-weights").textContent = "best.pt ready";
        document.getElementById("btn-infer").style.display = "";
      }
    } else {
      card("train", ""); badge("train", "idle", "idle"); numEl("train", "", 5);
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
    try {
      const s = await fetch("/status").then(r => r.json());
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

  // ── Init ──────────────────────────────────────────────────────────────────
  (async () => {
    const s = await poll();
    await loadRuns(s?.run_name || "");
    startPolling();
    if (s?.running) { window.loadLog(); }
  })();
})();
