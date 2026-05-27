"""Global object repository — stores curated crop PNGs indexed by Label + Identifier."""
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

from state import RUNS_DIR

REPO_DIR = RUNS_DIR / "_repo"


class RepoManager:
    def __init__(self, repo_dir: Path = REPO_DIR):
        self.repo_dir = repo_dir
        self.index_file = repo_dir / "repo_index.json"
        self.repo_dir.mkdir(parents=True, exist_ok=True)

    # ── Index ─────────────────────────────────────────────────────────────────

    def _load_index(self) -> dict:
        if not self.index_file.exists():
            return {"version": 1, "entries": {}}
        return json.loads(self.index_file.read_text())

    def _save_index(self, data: dict) -> None:
        tmp = self.index_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        os.replace(tmp, self.index_file)

    # ── Queries ───────────────────────────────────────────────────────────────

    def list_entries(self) -> dict:
        return self._load_index().get("entries", {})

    def entry_dir(self, label: str, identifier: str) -> Path:
        return self.repo_dir / label / identifier

    def _find_label_key(self, entries: dict, label: str):
        """Case-insensitive label lookup; returns the stored key or None."""
        for k in entries:
            if k.lower() == label.lower():
                return k
        return None

    # ── Mutations ─────────────────────────────────────────────────────────────

    def publish(
        self,
        label: str,
        identifier: str,
        source_dir: Path,
        source_run: str = None,
        notes: str = "",
        allow_append: bool = False,
    ) -> int:
        """Copy PNGs from source_dir to the repo. Returns image count added."""
        data = self._load_index()
        entries = data.setdefault("entries", {})

        label_key = self._find_label_key(entries, label)
        if label_key is None:
            label_key = label
            entries[label_key] = {"label": label_key, "identifiers": {}}

        label_entry = entries[label_key]
        identifiers = label_entry.setdefault("identifiers", {})

        if identifier in identifiers and not allow_append:
            raise ValueError(f"Identifier '{identifier}' already exists under '{label_key}'")

        dest_dir = self.repo_dir / label_key / identifier
        dest_dir.mkdir(parents=True, exist_ok=True)

        # Determine starting number for sequential rename
        existing = sorted(dest_dir.glob("*.png"))
        start_n = len(existing) + 1

        sources = sorted(source_dir.glob("*.png"))
        if not sources:
            raise ValueError(f"No PNG files found in {source_dir}")

        for i, src in enumerate(sources, start=start_n):
            shutil.copy2(src, dest_dir / f"{i:04d}.png")

        total = len(list(dest_dir.glob("*.png")))
        identifiers[identifier] = {
            "identifier": identifier,
            "image_count": total,
            "added_at": datetime.now().isoformat(timespec="seconds"),
            "source_run": source_run,
            "notes": notes,
        }
        self._save_index(data)
        return len(sources)


