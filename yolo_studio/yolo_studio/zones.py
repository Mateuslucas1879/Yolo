"""Zonas poligonais: ocupação, eventos de entrada/saída e desenho no quadro.

As zonas são guardadas em coordenadas NORMALIZADAS (0 a 1), então continuam
valendo se a resolução da câmera mudar.
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

from .detector import Detection


@dataclass
class Zone:
    name: str
    points: list  # [[x, y], ...] normalizados


@dataclass
class ZoneEvent:
    kind: str  # "enter" | "exit"
    zone: str
    track_id: int
    label: str


class ZoneManager:
    def __init__(self, exit_grace_frames: int = 8):
        self.exit_grace = exit_grace_frames
        self._zones: List[Zone] = []
        # nome da zona -> {track_id: [quadros_sem_ver, label]}
        self._state: Dict[str, Dict[int, list]] = {}
        self._occupancy: Dict[str, int] = {}
        self._lock = threading.RLock()

    # ---------------- edição ----------------
    def add(self, name: str, points: list) -> str:
        with self._lock:
            base, n = name.strip() or "Zona", 2
            final = base
            existing = {z.name for z in self._zones}
            while final in existing:
                final = f"{base} ({n})"
                n += 1
            self._zones.append(Zone(final, [list(map(float, p)) for p in points]))
            return final

    def clear(self) -> None:
        with self._lock:
            self._zones.clear()
            self._state.clear()
            self._occupancy.clear()

    def count(self) -> int:
        with self._lock:
            return len(self._zones)

    # ---------------- persistência ----------------
    def save(self, path: str) -> None:
        with self._lock:
            data = {"zones": [{"name": z.name, "points": z.points} for z in self._zones]}
        Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def load(self, path: str) -> None:
        p = Path(path)
        if not p.exists():
            return
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            with self._lock:
                self._zones = [Zone(z["name"], z["points"]) for z in data.get("zones", [])]
        except (json.JSONDecodeError, KeyError):
            pass  # arquivo inválido: começa sem zonas

    # ---------------- lógica ----------------
    @staticmethod
    def _poly(zone: Zone, w: int, h: int) -> np.ndarray:
        pts = np.array(zone.points, dtype=np.float32) * np.array([w, h], dtype=np.float32)
        return pts.reshape(-1, 1, 2)

    def update(self, detections: List[Detection], frame_shape) -> List[ZoneEvent]:
        """Atualiza ocupação e devolve eventos de entrada/saída deste quadro.

        O ponto de referência de cada detecção é o centro da borda inferior da
        caixa (os "pés" de uma pessoa), que representa melhor a posição no chão.
        """
        h, w = frame_shape[:2]
        events: List[ZoneEvent] = []

        with self._lock:
            for zone in self._zones:
                poly = self._poly(zone, w, h)
                inside: Dict[int, str] = {}
                n_inside = 0

                for d in detections:
                    x1, y1, x2, y2 = d.xyxy
                    point = (float((x1 + x2) / 2), float(y2))
                    if cv2.pointPolygonTest(poly, point, False) >= 0:
                        n_inside += 1
                        if d.track_id is not None:
                            inside[d.track_id] = d.label

                state = self._state.setdefault(zone.name, {})

                for tid, label in inside.items():
                    if tid not in state:
                        events.append(ZoneEvent("enter", zone.name, tid, label))
                    state[tid] = [0, label]

                for tid in list(state):
                    if tid in inside:
                        continue
                    state[tid][0] += 1
                    if state[tid][0] > self.exit_grace:
                        events.append(ZoneEvent("exit", zone.name, tid, state[tid][1]))
                        del state[tid]

                self._occupancy[zone.name] = n_inside

        return events

    # ---------------- desenho ----------------
    def draw(self, frame: np.ndarray) -> None:
        with self._lock:
            if not self._zones:
                return
            h, w = frame.shape[:2]
            overlay = frame.copy()
            drawn = []
            for zone in self._zones:
                occupied = self._occupancy.get(zone.name, 0) > 0
                color = (0, 0, 255) if occupied else (0, 200, 0)  # BGR
                poly = self._poly(zone, w, h).astype(np.int32)
                cv2.fillPoly(overlay, [poly], color)
                drawn.append((zone, poly, color))

            cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, dst=frame)

            for zone, poly, color in drawn:
                cv2.polylines(frame, [poly], True, color, 2)
                x, y = poly[0, 0]
                text = f"{zone.name}: {self._occupancy.get(zone.name, 0)}"
                cv2.putText(
                    frame, text, (int(x) + 4, int(y) + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA,
                )
