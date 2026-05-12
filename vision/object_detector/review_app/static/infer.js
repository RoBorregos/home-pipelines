const btnRun      = document.getElementById("btn-run");
const dropZone    = document.getElementById("drop-zone");
const status      = document.getElementById("status");
const imagesDiv   = document.getElementById("images");
const imgOrig     = document.getElementById("img-original");
const imgAnn      = document.getElementById("img-annotated");
const imgAnnPlaceholder = document.getElementById("img-annotated-placeholder");
const modelLabel  = document.getElementById("model-label");

let selectedFile = null;

// ── Init ──────────────────────────────────────────────────────────────────────

fetch("/status").then(r => r.json()).then(s => {
  if (s.best_weights) {
    const parts = s.best_weights.split("/");
    modelLabel.textContent = parts.slice(-4).join("/");
  } else {
    modelLabel.textContent = "No trained model";
  }
});

// ── Drag & drop ───────────────────────────────────────────────────────────────

dropZone.addEventListener("dragover", e => { e.preventDefault(); dropZone.classList.add("over"); });
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("over"));
dropZone.addEventListener("drop", e => {
  e.preventDefault();
  dropZone.classList.remove("over");
  const f = e.dataTransfer.files[0];
  if (f) onFileSelect(f);
});

function onFileSelect(file) {
  if (!file) return;
  selectedFile = file;
  dropZone.textContent = file.name;
  btnRun.disabled = false;
  const url = URL.createObjectURL(file);
  imgOrig.src = url;
  imgAnn.style.display = "none";
  imgAnnPlaceholder.style.display = "flex";
  imagesDiv.style.display = "grid";
  status.textContent = "";
  status.className = "";
}

// ── Inference ─────────────────────────────────────────────────────────────────

async function runInfer() {
  if (!selectedFile) return;
  const conf = document.getElementById("conf").value;

  btnRun.disabled = true;
  status.textContent = "Running…";
  status.className = "";

  const form = new FormData();
  form.append("file", selectedFile);

  try {
    const res = await fetch(`/infer?conf=${conf}`, {
      method: "POST",
      headers: { "x-api-key": getApiKey() },
      body: form,
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || res.statusText);
    }

    const blob = await res.blob();
    const url  = URL.createObjectURL(blob);
    imgAnn.src = url;
    imgAnn.style.display = "block";
    imgAnnPlaceholder.style.display = "none";
    status.textContent = "Done";
    status.className = "ok";
  } catch (e) {
    status.textContent = "Error: " + e.message;
    status.className = "err";
  } finally {
    btnRun.disabled = false;
  }
}
