"""Encapsula o modelo YOLO (detecção, rastreamento e pose)."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from ultralytics import YOLO

from .config import Config


@dataclass
class Detection:
    track_id: Optional[int]
    cls_id: int
    label: str
    conf: float
    xyxy: tuple  # (x1, y1, x2, y2) em pixels


@dataclass
class InferenceOutput:
    annotated: np.ndarray
    detections: List[Detection]
    infer_ms: float


def _resolve_device(pref: str) -> str:
    if pref != "auto":
        return pref
    try:
        import torch

        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


class Detector:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = _resolve_device(cfg.device)
        self.half = bool(cfg.half and self.device.startswith("cuda"))
        self.model: Optional[YOLO] = None
        self.model_name: Optional[str] = None

    # ------------------------------------------------------------------
    def ensure_model(self, name: str) -> bool:
        """Carrega o modelo se ainda não for o atual. Retorna True se carregou um novo."""
        if name == self.model_name:
            return False

        model = YOLO(name)
        # Aquecimento: a 1ª inferência é sempre mais lenta (alocação de memória/cuDNN)
        dummy = np.zeros((self.cfg.imgsz, self.cfg.imgsz, 3), dtype=np.uint8)
        model.predict(
            dummy,
            imgsz=self.cfg.imgsz,
            device=self.device,
            half=self.half,
            verbose=False,
        )
        self.model = model
        self.model_name = name
        return True

    # ------------------------------------------------------------------
    def infer(
        self,
        frame: np.ndarray,
        conf: float,
        iou: float,
        classes: Optional[list],
        tracking: bool,
    ) -> InferenceOutput:
        kwargs = dict(
            conf=conf,
            iou=iou,
            imgsz=self.cfg.imgsz,
            device=self.device,
            half=self.half,
            classes=classes,
            verbose=False,
        )

        t0 = time.perf_counter()
        if tracking:
            results = self.model.track(
                frame, persist=True, tracker=self.cfg.tracker, **kwargs
            )
        else:
            results = self.model.predict(frame, **kwargs)
        infer_ms = (time.perf_counter() - t0) * 1000.0

        r = results[0]
        detections: List[Detection] = []
        boxes = r.boxes
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy().astype(int)
            ids = (
                boxes.id.int().cpu().tolist()
                if boxes.id is not None
                else [None] * len(clss)
            )
            for box, c, k, tid in zip(xyxy, confs, clss, ids):
                detections.append(
                    Detection(
                        track_id=tid,
                        cls_id=int(k),
                        label=str(r.names[int(k)]),
                        conf=float(c),
                        xyxy=tuple(float(v) for v in box),
                    )
                )

        # r.plot() desenha caixas, IDs e (em modelos de pose) os esqueletos
        return InferenceOutput(annotated=r.plot(), detections=detections, infer_ms=infer_ms)
