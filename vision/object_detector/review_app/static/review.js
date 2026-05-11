(() => {
  const PAGE_SIZE = 48;
  let currentClass = "";
  let currentPage  = 0;
  let totalImages  = 0;
  const selected   = new Set();

  const grid      = document.getElementById("grid");
  const pagination = document.getElementById("pagination");
  const meta      = document.getElementById("header-meta");
  const countEl   = document.getElementById("selection-count");
  const btnDelete = document.getElementById("btn-delete");
  const btnApprove = document.getElementById("btn-approve");
  const btnSelectAll = document.getElementById("btn-select-all");
  const btnClear  = document.getElementById("btn-clear");
  const statusBar = document.getElementById("status-bar");

  // ── Helpers ─────────────────────────────────────────────────────────────────

  function showStatus(msg, type = "info") {
    statusBar.textContent = msg;
    statusBar.className = `visible ${type}`;
    if (type !== "error") setTimeout(() => { statusBar.className = ""; }, 3000);
  }

  function updateSelectionUI() {
    const n = selected.size;
    countEl.textContent = n > 0 ? `${n} selected` : "";
    btnDelete.disabled = n === 0 || !currentClass;
    grid.querySelectorAll(".card").forEach(card => {
      card.classList.toggle("selected", selected.has(card.dataset.path));
    });
  }

  // ── Class selector ───────────────────────────────────────────────────────────

  async function loadClasses() {
    const s = await fetch("/status").then(r => r.json());
    const classes = Object.keys(s.segmented_classes || {});
    const bar = document.getElementById("class-bar");
    bar.innerHTML = "";

    if (!classes.length) {
      bar.innerHTML = '<span style="color:#555;font-size:0.8rem">No segmented classes yet — run the segment stage first.</span>';
      return;
    }

    classes.forEach(cls => {
      const btn = document.createElement("button");
      btn.className = "class-btn";
      btn.textContent = cls;
      btn.addEventListener("click", () => selectClass(cls));
      bar.appendChild(btn);
    });

    // Auto-select first class
    selectClass(classes[0]);
  }

  function selectClass(cls) {
    currentClass = cls;
    currentPage = 0;
    selected.clear();

    document.getElementById("hdr-class").textContent = cls;
    document.querySelectorAll(".class-btn").forEach(b => {
      b.classList.toggle("active", b.textContent === cls);
    });

    btnSelectAll.disabled = false;
    btnClear.disabled = false;
    btnApprove.disabled = false;

    loadPage(0);
  }

  // ── Grid rendering ───────────────────────────────────────────────────────────

  async function loadPage(page) {
    if (!currentClass) return;
    const res = await fetch(`/review/images?class_name=${encodeURIComponent(currentClass)}&page=${page}&page_size=${PAGE_SIZE}`);
    const data = await res.json();
    totalImages = data.total;
    currentPage = page;

    meta.textContent = `${totalImages} images`;

    if (!data.images.length) {
      grid.innerHTML = '<p class="empty">No images for this class.</p>';
      pagination.innerHTML = "";
      return;
    }

    grid.innerHTML = data.images.map(path => `
      <div class="card" data-path="${path}">
        <img src="${path}" loading="lazy" />
        <div class="check"></div>
      </div>
    `).join("");

    grid.querySelectorAll(".card").forEach(card => {
      card.addEventListener("click", () => {
        const p = card.dataset.path;
        selected.has(p) ? selected.delete(p) : selected.add(p);
        updateSelectionUI();
      });
    });

    renderPagination();
    updateSelectionUI();
  }

  function renderPagination() {
    const totalPages = Math.ceil(totalImages / PAGE_SIZE);
    if (totalPages <= 1) { pagination.innerHTML = ""; return; }
    pagination.innerHTML = `
      <button class="btn-ghost" ${currentPage === 0 ? "disabled" : ""} id="pg-prev">← Prev</button>
      <span>Page ${currentPage + 1} / ${totalPages}</span>
      <button class="btn-ghost" ${currentPage >= totalPages - 1 ? "disabled" : ""} id="pg-next">Next →</button>
    `;
    document.getElementById("pg-prev")?.addEventListener("click", () => loadPage(currentPage - 1));
    document.getElementById("pg-next")?.addEventListener("click", () => loadPage(currentPage + 1));
  }

  // ── Actions ──────────────────────────────────────────────────────────────────

  btnSelectAll.addEventListener("click", () => {
    grid.querySelectorAll(".card").forEach(c => selected.add(c.dataset.path));
    updateSelectionUI();
  });

  btnClear.addEventListener("click", () => {
    selected.clear();
    updateSelectionUI();
  });

  btnDelete.addEventListener("click", async () => {
    if (!selected.size) return;
    btnDelete.disabled = true;
    showStatus(`Deleting ${selected.size} images…`);
    const res = await fetch("/review/delete", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ paths: [...selected] }),
    });
    const data = await res.json();
    selected.clear();
    showStatus(`Deleted ${data.deleted} images`, "success");
    await loadPage(currentPage);
  });

  btnApprove.addEventListener("click", async () => {
    if (!confirm("Mark all classes as reviewed? This enables the Generate stage.")) return;
    const res = await fetch("/review/approve", {
      method: "POST",
      headers: { "x-api-key": getApiKey() },
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      showStatus(`Error: ${err.detail || res.statusText}`, "error");
      return;
    }
    showStatus("Marked as reviewed. You can now run Generate from the dashboard.", "success");
    btnApprove.disabled = true;
  });

  // ── Init ─────────────────────────────────────────────────────────────────────
  loadClasses();
})();
