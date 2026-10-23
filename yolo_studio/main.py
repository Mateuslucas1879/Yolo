"""Ponto de entrada: python main.py [--config config.yaml] [--source 0|video.mp4]"""
import argparse
import tkinter as tk

from yolo_studio.config import Config
from yolo_studio.ui import App


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO Studio")
    parser.add_argument("--config", default="config.yaml", help="caminho do config.yaml")
    parser.add_argument("--source", default=None,
                        help="sobrescreve a fonte: índice da câmera (0, 1...) ou caminho de vídeo")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    if args.source is not None:
        cfg.source = int(args.source) if args.source.isdigit() else args.source

    root = tk.Tk()
    App(root, cfg)
    root.mainloop()


if __name__ == "__main__":
    main()
