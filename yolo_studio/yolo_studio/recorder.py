"""Gravação de vídeo com o FPS real do processamento."""
from __future__ import annotations

import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


class Recorder:
    def __init__(self, out_dir: str):
        self.out_dir = Path(out_dir)
        self._writer: Optional[cv2.VideoWriter] = None
        self._path: Optional[Path] = None
        self._lock = threading.Lock()

    @property
    def active(self) -> bool:
        return self._writer is not None

    def start(self, width: int, height: int, fps: float) -> Optional[Path]:
        with self._lock:
            if self._writer is not None:
                return self._path
            self.out_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = self.out_dir / f"yolo_record_{stamp}.mp4"
            writer = cv2.VideoWriter(
                str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height)
            )
            if not writer.isOpened():
                return None
            self._writer, self._path = writer, path
            return path

    def write(self, frame: np.ndarray) -> None:
        with self._lock:
            if self._writer is not None:
                self._writer.write(frame)

    def stop(self) -> Optional[Path]:
        with self._lock:
            if self._writer is None:
                return None
            self._writer.release()
            path, self._writer, self._path = self._path, None, None
            return path
