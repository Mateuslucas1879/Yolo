"""Pipeline: captura -> YOLO -> zonas -> desenho -> gravação.

Roda numa thread própria, então a interface nunca trava durante a inferência.
A interface só lê o resultado mais recente (`latest()`) e os avisos (`event_queue`).
"""
from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np

from .capture import FrameSource
from .config import Config, RuntimeSettings
from .detector import Detector
from .events import EventLogger
from .recorder import Recorder
from .zones import ZoneManager


@dataclass
class FrameResult:
    seq: int
    frame: np.ndarray
    n_objects: int
    class_counts: Dict[str, int] = field(default_factory=dict)
    fps: float = 0.0
    infer_ms: float = 0.0


class Pipeline:
    def __init__(
        self,
        cfg: Config,
        settings: RuntimeSettings,
        detector: Detector,
        zones: ZoneManager,
        events: EventLogger,
    ):
        self.cfg = cfg
        self.settings = settings
        self.detector = detector
        self.zones = zones
        self.events = events
        self.recorder = Recorder(cfg.recordings_dir)

        self.event_queue: "queue.Queue[str]" = queue.Queue()
        self.status = "Parado"
        self.fps = 0.0

        self._source: Optional[FrameSource] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest: Optional[FrameResult] = None
        self._seq = 0

    # ---------------- estado ----------------
    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def recording(self) -> bool:
        return self.recorder.active

    def latest(self) -> Optional[FrameResult]:
        with self._lock:
            return self._latest

    def _emit(self, msg: str) -> None:
        """Aviso para a lista de alertas da interface."""
        self.event_queue.put(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

    # ---------------- ciclo de vida ----------------
    def start(self) -> bool:
        if self.running:
            return True
        source = FrameSource(self.cfg.source, self.cfg.frame_width, self.cfg.frame_height)
        if not source.open():
            return False

        self._source = source
        self._stop.clear()
        self._seq = 0
        self._latest = None
        self.fps = 0.0
        self.status = "Iniciando…"
        self.events.log("start", detail=str(self.cfg.source))
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=15)
            self._thread = None
        self.status = "Parado"

    # ---------------- gravação / foto ----------------
    def start_recording(self) -> Optional[Path]:
        res = self.latest()
        if res is None or not self.running:
            return None
        h, w = res.frame.shape[:2]
        fps = self.fps if self.fps >= 1.0 else 15.0
        path = self.recorder.start(w, h, fps)
        if path:
            self.events.log("record_start", detail=str(path))
            self._emit(f"Gravando: {path.name}")
        return path

    def stop_recording(self) -> Optional[Path]:
        path = self.recorder.stop()
        if path:
            self.events.log("record_stop", detail=str(path))
            self._emit(f"Gravação salva: {path.name}")
        return path

    def snapshot(self) -> Optional[Path]:
        res = self.latest()
        if res is None:
            return None
        out = Path(self.cfg.snapshots_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        cv2.imwrite(str(path), res.frame)
        self.events.log("snapshot", detail=str(path))
        self._emit(f"Foto salva: {path.name}")
        return path

    # ---------------- thread de trabalho ----------------
    def _run(self) -> None:
        source = self._source
        known_tracks: set = set()
        last_t = time.perf_counter()
        fps_ema = 0.0
        stale_reported = False

        try:
            while not self._stop.is_set():
                s = self.settings

                # Troca de modelo pedida pela interface (carrega aqui, fora da UI)
                if self.detector.model_name != s.model_name:
                    self.status = f"Carregando {s.model_name}…"
                    self._emit(f"Carregando modelo {s.model_name}…")
                    try:
                        self.detector.ensure_model(s.model_name)
                        self.events.log("model_loaded", detail=s.model_name)
                        known_tracks.clear()
                    except Exception as exc:  # download falhou, arquivo inexistente...
                        self._emit(f"Falha ao carregar {s.model_name}: {exc}")
                        if self.detector.model_name is None:
                            raise
                        s.model_name = self.detector.model_name

                frame = source.grab()

                if frame is None:
                    if source.finished:
                        self._emit("Fim do vídeo.")
                        self.status = "Fim do vídeo"
                        break
                    if (not source.is_file
                            and source.age() > self.cfg.stale_timeout_s
                            and not stale_reported):
                        stale_reported = True
                        self.status = "SEM IMAGEM"
                        self._emit("⚠ SEM IMAGEM da câmera!")
                        self.events.log("camera_lost")
                    time.sleep(0.005)
                    continue

                if stale_reported:
                    stale_reported = False
                    self._emit("Câmera voltou a enviar imagem.")
                    self.events.log("camera_restored")
                self.status = "Ativo"

                # --- inferência ---
                out = self.detector.infer(frame, s.conf, s.iou, s.classes, s.tracking)

                # --- novos IDs ---
                if self.cfg.log_new_tracks:
                    for d in out.detections:
                        if d.track_id is not None and d.track_id not in known_tracks:
                            known_tracks.add(d.track_id)
                            self.events.log("new_track", track_id=d.track_id,
                                            label=d.label, conf=d.conf)

                # --- zonas ---
                for ev in self.zones.update(out.detections, out.annotated.shape):
                    verb = "ENTROU em" if ev.kind == "enter" else "saiu de"
                    icon = "⚠" if ev.kind == "enter" else "•"
                    self._emit(f"{icon} {ev.label} #{ev.track_id} {verb} {ev.zone}")
                    self.events.log(f"zone_{ev.kind}", zone=ev.zone,
                                    track_id=ev.track_id, label=ev.label)
                self.zones.draw(out.annotated)

                # --- gravação (já com caixas e zonas desenhadas) ---
                self.recorder.write(out.annotated)

                # --- FPS (média móvel) ---
                now = time.perf_counter()
                inst = 1.0 / max(now - last_t, 1e-6)
                last_t = now
                fps_ema = inst if fps_ema == 0.0 else 0.9 * fps_ema + 0.1 * inst
                self.fps = fps_ema

                counts: Dict[str, int] = {}
                for d in out.detections:
                    counts[d.label] = counts.get(d.label, 0) + 1

                self._seq += 1
                result = FrameResult(
                    seq=self._seq,
                    frame=out.annotated,
                    n_objects=len(out.detections),
                    class_counts=counts,
                    fps=fps_ema,
                    infer_ms=out.infer_ms,
                )
                with self._lock:
                    self._latest = result

        except Exception as exc:
            self.status = f"Erro: {exc}"
            self._emit(f"ERRO: {exc}")
            self.events.log("error", detail=repr(exc))
        finally:
            if self.recorder.active:
                self.stop_recording()
            source.release()
            self.events.log("stop")
