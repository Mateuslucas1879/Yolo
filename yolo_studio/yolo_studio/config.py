"""Carregamento da configuração (config.yaml) e parâmetros ajustáveis em tempo real."""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Optional, Union

import yaml


@dataclass
class Config:
    source: Union[int, str] = 0
    frame_width: int = 1280
    frame_height: int = 720

    models: list = field(default_factory=lambda: ["yolo11m.pt"])
    default_model: str = "yolo11m.pt"

    imgsz: int = 960
    conf: float = 0.5
    iou: float = 0.5
    device: str = "auto"
    half: bool = True

    tracking: bool = True
    tracker: str = "bytetrack.yaml"

    class_presets: dict = field(default_factory=lambda: {"Todas": None, "Pessoas": [0]})

    zones_file: str = "zones.json"
    zone_exit_grace_frames: int = 8

    stale_timeout_s: float = 2.0

    log_new_tracks: bool = True
    logs_dir: str = "logs"
    recordings_dir: str = "recordings"
    snapshots_dir: str = "snapshots"

    @classmethod
    def load(cls, path: str | Path = "config.yaml") -> "Config":
        path = Path(path)
        data = {}
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}

        valid = {f.name for f in fields(cls)}
        cfg = cls(**{k: v for k, v in data.items() if k in valid})

        # "0" -> 0 (índice de câmera); qualquer outra string é caminho de vídeo
        if isinstance(cfg.source, str) and cfg.source.strip().isdigit():
            cfg.source = int(cfg.source.strip())

        if cfg.default_model not in cfg.models:
            cfg.models.insert(0, cfg.default_model)
        if not cfg.class_presets:
            cfg.class_presets = {"Todas": None}
        return cfg


@dataclass
class RuntimeSettings:
    """Parâmetros que a interface altera enquanto o pipeline está rodando.

    A interface escreve, a thread de inferência lê a cada quadro.
    (Atribuições simples de atributos são seguras entre threads no CPython.)
    """

    model_name: str
    conf: float
    iou: float
    tracking: bool
    classes: Optional[list] = None
