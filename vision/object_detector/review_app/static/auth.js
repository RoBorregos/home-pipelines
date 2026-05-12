(() => {
  const STORAGE_KEY = "od_api_key";

  function _showModal() {
    const overlay = document.createElement("div");
    overlay.id = "auth-overlay";
    overlay.style.cssText =
      "position:fixed;inset:0;background:#000c;display:flex;align-items:center;" +
      "justify-content:center;z-index:9999;font-family:system-ui,sans-serif";

    overlay.innerHTML = `
      <div style="background:#1a1a1a;border:1px solid #2a2a2a;border-radius:12px;
                  padding:28px;width:min(340px,calc(100vw - 32px))">
        <h2 style="font-size:0.95rem;font-weight:700;color:#e0e0e0;margin:0 0 6px">API Key</h2>
        <p style="font-size:0.76rem;color:#666;margin:0 0 16px">Required to control the pipeline.</p>
        <input id="auth-input" type="text" placeholder="your-api-key"
          autocomplete="off" spellcheck="false"
          style="width:100%;background:#0f0f0f;border:1px solid #2a2a2a;color:#e0e0e0;
                 border-radius:6px;padding:8px 12px;font-size:0.84rem;font-family:monospace;
                 margin-bottom:12px;box-sizing:border-box;outline:none" />
        <p id="auth-err"
          style="font-size:0.72rem;color:#f87171;margin:0 0 12px;display:none">
          Please enter a key.
        </p>
        <div style="display:flex;justify-content:flex-end">
          <button id="auth-save"
            style="background:#2563eb;color:#fff;border:none;border-radius:6px;
                   padding:8px 18px;font-size:0.84rem;font-weight:600;cursor:pointer">
            Save
          </button>
        </div>
      </div>
    `;

    document.body.appendChild(overlay);

    const input = document.getElementById("auth-input");
    const btn   = document.getElementById("auth-save");
    const err   = document.getElementById("auth-err");

    input.value = localStorage.getItem(STORAGE_KEY) || "";
    setTimeout(() => input.focus(), 50);

    function commit() {
      const k = input.value.trim();
      if (!k) { err.style.display = "block"; return; }
      localStorage.setItem(STORAGE_KEY, k);
      overlay.remove();
    }

    btn.addEventListener("click", commit);
    input.addEventListener("keydown", e => { if (e.key === "Enter") commit(); });
  }

  window.getApiKey   = () => localStorage.getItem(STORAGE_KEY) || "";
  window.resetApiKey = () => _showModal();

  document.addEventListener("DOMContentLoaded", () => {
    if (!localStorage.getItem(STORAGE_KEY)) _showModal();
  });
})();
