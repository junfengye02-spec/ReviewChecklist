from __future__ import annotations

import json
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any


class RunArtifacts:
    def __init__(self, runs_dir: Path, run_id: str | None = None) -> None:
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = runs_dir / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=False)

    @classmethod
    def open_existing(cls, run_dir: Path) -> "RunArtifacts":
        resolved = run_dir.resolve()
        if not resolved.is_dir():
            raise FileNotFoundError(f"Run directory not found: {resolved}")
        instance = cls.__new__(cls)
        instance.run_id = resolved.name
        instance.run_dir = resolved
        return instance

    def read_json(self, name: str) -> Any:
        return json.loads((self.run_dir / name).read_text(encoding="utf-8"))

    def write_json(self, name: str, value: Any) -> Path:
        path = self.run_dir / name
        path.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        return path

    def write_text(self, name: str, value: str) -> Path:
        path = self.run_dir / name
        path.write_text(value, encoding="utf-8")
        return path

    def copy_excel(self, source: Path) -> Path:
        target = self.run_dir / "审核要点_原始.xlsx"
        shutil.copy2(source, target)
        return target

    def build_zip(self, pdf_path: Path) -> Path:
        target = self.run_dir / "待审核材料_MVP.zip"
        archive_name = f"待审核材料/{pdf_path.name}"
        with zipfile.ZipFile(
            target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
        ) as archive:
            archive.write(pdf_path, arcname=archive_name)
        return target
