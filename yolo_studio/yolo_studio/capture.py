"""Captura de vídeo (câmera ou arquivo).

- Câmera: uma thread lê continuamente e guarda só o quadro mais recente.
  Se a inferência for mais lenta que a câmera, quadros antigos são descartados
  (nada de atraso acumulando).
- Arquivo: lê quadro a quadro, sem descartar, para testes reproduzíveis.
"""
from __future__ import annotations

import os
import threading
import time
from typing import Optional, Union

import cv2
import numpy as np


class FrameSource:
    def __init__(self, source: Union[int, str], width: int = 1280, height: int = 720):
        self.source = source
        self.width = width
        self.height = height
        self.is_file = isinstance(source, str)
        self.finished = False

        self._cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._seq = 0
        self._returned_seq = 0
        self._last_ok = time.monotonic()

    # ------------------------------------------------------------------
    def open(self) -> bool:
        if self.is_file:
            self._cap = cv2.VideoCapture(self.source)
        elif os.name == "nt":
            self._cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
        else:
            self._cap = cv2.VideoCapture(self.source)

        if not self._cap.isOpened():
            self._cap.release()
            self._cap = None
            return False

        if not self.is_file:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

        self._last_ok = time.monotonic()
        return True

    def _loop(self) -> None:
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if ok:
                with self._lock:
                    self._frame = frame
                    self._seq += 1
                    self._last_ok = time.monotonic()
            else:
                time.sleep(0.05)

    # ------------------------------------------------------------------
    def grab(self) -> Optional[np.ndarray]:
        """Devolve um quadro novo, ou None se ainda não há (ou se o vídeo acabou)."""
        if self.is_file:
            ok, frame = self._cap.read()
            if not ok:
                self.finished = True
                return None
            self._last_ok = time.monotonic()
            return frame

        with self._lock:
            if self._seq == self._returned_seq:
                return None
            self._returned_seq = self._seq
            return self._frame

    def age(self) -> float:
        """Segundos desde o último quadro recebido com sucesso."""
        return time.monotonic() - self._last_ok

    def release(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        if self._cap is not None:
            self._cap.release()
            self._cap = None
