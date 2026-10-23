"""Log de eventos em CSV (um arquivo por execução, criado no primeiro evento)."""
from __future__ import annotations

import csv
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

HEADER = ["timestamp", "evento", "zona", "track_id", "classe", "confianca", "detalhe"]


class EventLogger:
    def __init__(self, logs_dir: str):
        self.logs_dir = Path(logs_dir)
        self.path: Optional[Path] = None
        self._file = None
        self._writer = None
        self._lock = threading.Lock()

    def _open(self) -> None:
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = self.logs_dir / f"events_{stamp}.csv"
        self._file = open(self.path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow(HEADER)

    def log(self, event: str, zone: str = "", track_id="", label: str = "",
            conf="", detail: str = "") -> None:
        with self._lock:
            if self._writer is None:
                self._open()
            conf_txt = f"{conf:.3f}" if isinstance(conf, float) else conf
            self._writer.writerow([
                datetime.now().isoformat(timespec="milliseconds"),
                event, zone, track_id, label, conf_txt, detail,
            ])
            self._file.flush()  # não perde dados se o programa fechar de repente

    def close(self) -> None:
        with self._lock:
            if self._file is not None:
                self._file.close()
                self._file = None
                self._writer = None
