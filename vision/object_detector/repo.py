"""Global object repository — stores curated crop PNGs indexed by Label + Identifier."""
import json
import logging
import os
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

from state import RUNS_DIR

REPO_DIR   = RUNS_DIR / "_repo"
REPO_INDEX = REPO_DIR / "repo_index.json"

_EMPTY_INDEX = {"version": 1, "entries": {}}


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

    def list_labels(self) -> list:
        return sorted(self.list_entries().keys())

    def list_identifiers(self, label: str) -> list:
        entries = self.list_entries()
        label_key = self._find_label_key(entries, label)
        if label_key is None:
            return []
        return sorted(entries[label_key].get("identifiers", {}).keys())

    def entry_dir(self, label: str, identifier: str) -> Path:
        return self.repo_dir / label / identifier

    def _find_label_key(self, entries: dict, label: str):
        """Case-insensitive label lookup; returns the stored key or None."""
        for k in entries:
            if k.lower() == label.lower():
                return k
        return None

    def validate_entry(self, label: str, identifier: str) -> dict:
        entries = self.list_entries()
        label_key = self._find_label_key(entries, label)
        if label_key is None:
            return {"ok": False, "image_count": 0, "index_count": 0, "drift": 0, "missing": True}
        id_data = entries[label_key].get("identifiers", {}).get(identifier)
        if id_data is None:
            return {"ok": False, "image_count": 0, "index_count": 0, "drift": 0, "missing": True}
        entry_dir = self.entry_dir(label_key, identifier)
        actual = len(list(entry_dir.glob("*.png"))) if entry_dir.exists() else 0
        index_count = id_data.get("image_count", 0)
        return {
            "ok": actual == index_count and actual > 0,
            "image_count": actual,
            "index_count": index_count,
            "drift": actual - index_count,
            "missing": not entry_dir.exists(),
        }

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

    def import_to_run(self, label: str, identifier: str, run_cropped_dir: Path) -> int:
        """Copy repo entry PNGs into the run's cropped/{label}/ directory."""
        entries = self.list_entries()
        label_key = self._find_label_key(entries, label)
        if label_key is None:
            raise ValueError(f"Label '{label}' not found in repository")
        if identifier not in entries[label_key].get("identifiers", {}):
            raise ValueError(f"Identifier '{identifier}' not found under '{label_key}'")

        src_dir = self.repo_dir / label_key / identifier
        if not src_dir.exists():
            raise ValueError(f"Repository directory missing: {src_dir}")

        dest_dir = run_cropped_dir / label_key
        dest_dir.mkdir(parents=True, exist_ok=True)

        sources = sorted(src_dir.glob("*.png"))
        for src in sources:
            shutil.copy2(src, dest_dir / src.name)

        return len(sources)

    def delete_entry(self, label: str, identifier: str) -> int:
        """Remove an identifier from the repo. Returns deleted image count."""
        data = self._load_index()
        entries = data.get("entries", {})
        label_key = self._find_label_key(entries, label)
        if label_key is None:
            return 0
        id_data = entries[label_key].get("identifiers", {}).pop(identifier, None)
        if id_data is None:
            return 0
        if not entries[label_key]["identifiers"]:
            del entries[label_key]
        self._save_index(data)
        entry_dir = self.repo_dir / label_key / identifier
        deleted = 0
        if entry_dir.exists():
            deleted = len(list(entry_dir.glob("*.png")))
            shutil.rmtree(entry_dir)
        return deleted

    def import_from_gdrive(
        self,
        drive_url: str,
        label: str,
        identifier: str,
        log: logging.Logger,
        notes: str = "",
    ) -> int:
        """Download a Drive folder of PNGs and publish them to the repo."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            log.info("Downloading Drive folder for repo import: %s", drive_url)
            result = subprocess.run(
                ["gdown", "--folder", drive_url, "-O", str(tmp_path), "--quiet"],
                capture_output=True, text=True,
            )
            if result.returncode != 0:
                log.error("gdown failed:\n%s", result.stderr)
                raise RuntimeError(f"gdown failed: {result.stderr[:300]}")

            roots = [d for d in tmp_path.iterdir() if d.is_dir()]
            download_root = roots[0] if len(roots) == 1 else tmp_path

            # Collect PNGs recursively (user may have a single flat folder)
            all_pngs = sorted(download_root.rglob("*.png"))
            if not all_pngs:
                raise RuntimeError("No PNG files found in downloaded Drive folder")

            log.info("Found %d PNG(s) — publishing to repo as %s / %s", len(all_pngs), label, identifier)

            # Copy all pngs to a temp flat dir for publish()
            flat_dir = tmp_path / "_flat"
            flat_dir.mkdir()
            for p in all_pngs:
                shutil.copy2(p, flat_dir / p.name)

            count = self.publish(label, identifier, flat_dir, source_run=None, notes=notes or "imported from Drive")
            log.info("Repo import done: %d images → %s / %s", count, label, identifier)
            return count
