"""Interface Tkinter. Só desenha e repassa comandos: nenhuma inferência roda aqui."""
from __future__ import annotations

import queue
import tkinter as tk
from tkinter import messagebox, simpledialog, ttk

import cv2
from PIL import Image, ImageTk

from .config import Config, RuntimeSettings
from .detector import Detector
from .events import EventLogger
from .pipeline import FrameResult, Pipeline
from .zones import ZoneManager

# Tema
BG = "#121212"
PANEL = "#1e1e1e"
CARD = "#2a2a2a"
ACCENT = "#00adb5"
TEXT = "#eeeeee"
GREEN = "#2e7d32"
RED = "#c62828"
ORANGE = "#ef6c00"
FONT = "Segoe UI"


class App:
    def __init__(self, root: tk.Tk, cfg: Config):
        self.root = root
        self.cfg = cfg

        # ---- núcleo (sem nada de interface) ----
        first_preset = next(iter(cfg.class_presets))
        self.settings = RuntimeSettings(
            model_name=cfg.default_model,
            conf=cfg.conf,
            iou=cfg.iou,
            tracking=cfg.tracking,
            classes=cfg.class_presets[first_preset],
        )
        self.events = EventLogger(cfg.logs_dir)
        self.zones = ZoneManager(cfg.zone_exit_grace_frames)
        self.zones.load(cfg.zones_file)
        self.detector = Detector(cfg)
        self.pipeline = Pipeline(cfg, self.settings, self.detector, self.zones, self.events)

        # ---- estado da interface ----
        self._last_seq = -1
        self._photo = None
        self._img_item = None
        self._ox = self._oy = self._dw = self._dh = 0  # posição/tamanho da imagem no canvas
        self._ui_running = False
        self._ui_recording = False
        self._drawing = False
        self._draft: list = []  # pontos da zona em desenho (normalizados)
        self._closing = False

        root.title("YOLO Studio")
        root.geometry("1320x820")
        root.minsize(1100, 700)
        root.configure(bg=BG)
        self._setup_style()
        self._build()

        root.bind("<Escape>", lambda e: self._cancel_zone())
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        self._poll()

    # ------------------------------------------------------------------
    # Construção da interface
    # ------------------------------------------------------------------
    def _setup_style(self) -> None:
        st = ttk.Style()
        st.theme_use("clam")
        st.configure("TScale", background=PANEL, troughcolor="#333333", sliderlength=15)
        st.configure("TCombobox", fieldbackground="#333333", background="#333333",
                     foreground=TEXT, arrowcolor=TEXT)
        st.map("TCombobox", fieldbackground=[("readonly", "#333333")],
               foreground=[("readonly", TEXT)])
        st.configure("TCheckbutton", background=PANEL, foreground=TEXT)
        self.root.option_add("*TCombobox*Listbox.background", "#333333")
        self.root.option_add("*TCombobox*Listbox.foreground", TEXT)

    def _label(self, parent, text, **kw):
        return tk.Label(parent, text=text, bg=kw.pop("bg", PANEL),
                        fg=kw.pop("fg", TEXT), font=kw.pop("font", (FONT, 9)), **kw)

    def _section(self, parent, text):
        self._label(parent, text, fg=ACCENT, font=(FONT, 10, "bold")).pack(
            anchor="w", padx=12, pady=(10, 3))

    def _button(self, parent, text, cmd, bg="#333333", fg="white", state=tk.NORMAL):
        b = tk.Button(parent, text=text, command=cmd, bg=bg, fg=fg, relief=tk.FLAT,
                      font=(FONT, 10, "bold"), cursor="hand2", state=state,
                      activebackground="#444444", activeforeground="white")
        b.pack(fill=tk.X, padx=12, pady=3)
        return b

    def _build(self) -> None:
        # Cabeçalho
        header = tk.Frame(self.root, bg=PANEL, height=44)
        header.pack(fill=tk.X, side=tk.TOP)
        self._label(header, "🔍 YOLO Studio — Detecção, Rastreamento e Zonas",
                    fg=ACCENT, font=(FONT, 13, "bold")).pack(side=tk.LEFT, padx=16, pady=8)
        self.lbl_rec = self._label(header, "", fg=RED, font=(FONT, 11, "bold"))
        self.lbl_rec.pack(side=tk.RIGHT, padx=16)

        main = tk.Frame(self.root, bg=BG)
        main.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

        # Vídeo (canvas, para poder desenhar zonas com o mouse)
        vframe = tk.Frame(main, bg="black", bd=2, relief=tk.SOLID)
        vframe.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(vframe, bg="black", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self._on_canvas_left)
        self.canvas.bind("<Button-3>", lambda e: self._finish_zone())
        self.canvas.bind("<Button-2>", lambda e: self._finish_zone())  # macOS

        # Painel de controles
        panel = tk.Frame(main, bg=PANEL, width=310)
        panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(12, 0))
        panel.pack_propagate(False)

        self._section(panel, "CONTROLES")
        self.btn_toggle = self._button(panel, "▶ Ligar", self.toggle_camera, bg=GREEN)
        self.btn_record = self._button(panel, "⏺ Iniciar gravação", self.toggle_recording,
                                       bg=RED, state=tk.DISABLED)
        self.btn_snap = self._button(panel, "📸 Capturar foto", self.take_snapshot,
                                     state=tk.DISABLED)

        self._section(panel, "MODELO")
        self.cmb_model = ttk.Combobox(panel, values=self.cfg.models, state="readonly")
        self.cmb_model.set(self.settings.model_name)
        self.cmb_model.pack(fill=tk.X, padx=12, pady=2)
        self.cmb_model.bind("<<ComboboxSelected>>", self._on_model)

        self._label(panel, "Filtro de classes:").pack(anchor="w", padx=12, pady=(6, 0))
        self.cmb_classes = ttk.Combobox(panel, values=list(self.cfg.class_presets),
                                        state="readonly")
        self.cmb_classes.current(0)
        self.cmb_classes.pack(fill=tk.X, padx=12, pady=2)
        self.cmb_classes.bind("<<ComboboxSelected>>", self._on_classes)

        self.lbl_conf = self._label(panel, "")
        self.lbl_conf.pack(anchor="w", padx=12, pady=(8, 0))
        self.sld_conf = ttk.Scale(panel, from_=0.05, to=1.0, value=self.settings.conf,
                                  orient=tk.HORIZONTAL, command=self._on_conf)
        self.sld_conf.pack(fill=tk.X, padx=12, pady=2)

        self.lbl_iou = self._label(panel, "")
        self.lbl_iou.pack(anchor="w", padx=12, pady=(4, 0))
        self.sld_iou = ttk.Scale(panel, from_=0.1, to=0.95, value=self.settings.iou,
                                 orient=tk.HORIZONTAL, command=self._on_iou)
        self.sld_iou.pack(fill=tk.X, padx=12, pady=2)
        self._on_conf(self.settings.conf)
        self._on_iou(self.settings.iou)

        self.var_track = tk.BooleanVar(value=self.settings.tracking)
        ttk.Checkbutton(panel, text="Rastrear (IDs persistentes)", variable=self.var_track,
                        command=self._on_tracking).pack(anchor="w", padx=12, pady=(6, 0))

        self._section(panel, "ZONAS")
        self.btn_zone = self._button(panel, "✏ Desenhar zona", self._toggle_zone_drawing)
        self._button(panel, "🗑 Limpar zonas", self._clear_zones)
        self.lbl_zone_info = self._label(panel, "", fg="#aaaaaa", wraplength=280, justify="left")
        self.lbl_zone_info.pack(anchor="w", padx=12)
        self._update_zone_info()

        self._section(panel, "ESTATÍSTICAS")
        card = tk.Frame(panel, bg=CARD, bd=1, relief=tk.SOLID)
        card.pack(fill=tk.X, padx=12, pady=3)
        self.lbl_count = self._label(card, "Objetos: 0", bg=CARD, font=(FONT, 12, "bold"))
        self.lbl_count.pack(anchor="w", padx=8, pady=(4, 0))
        self.lbl_fps = self._label(card, "0 FPS · 0 ms", bg=CARD, fg=ACCENT,
                                   font=(FONT, 11, "bold"))
        self.lbl_fps.pack(anchor="w", padx=8)
        self.lbl_classes = self._label(card, "", bg=CARD, fg="#aaaaaa", wraplength=270,
                                       justify="left")
        self.lbl_classes.pack(anchor="w", padx=8, pady=(0, 4))
        self.lbl_status = self._label(panel, "Status: Parado", font=(FONT, 9, "bold"))
        self.lbl_status.pack(anchor="w", padx=12, pady=(4, 0))

        self._section(panel, "ALERTAS")
        self.lst_alerts = tk.Listbox(panel, height=6, bg=CARD, fg=TEXT, relief=tk.FLAT,
                                     font=(FONT, 8), highlightthickness=0, activestyle="none")
        self.lst_alerts.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 12))

    # ------------------------------------------------------------------
    # Callbacks de parâmetros
    # ------------------------------------------------------------------
    def _on_model(self, _e=None) -> None:
        self.settings.model_name = self.cmb_model.get()

    def _on_classes(self, _e=None) -> None:
        self.settings.classes = self.cfg.class_presets[self.cmb_classes.get()]

    def _on_conf(self, val) -> None:
        self.settings.conf = float(val)
        self.lbl_conf.config(text=f"Confiança mínima: {int(float(val) * 100)}%")

    def _on_iou(self, val) -> None:
        self.settings.iou = float(val)
        self.lbl_iou.config(text=f"IoU (NMS): {float(val):.2f}")

    def _on_tracking(self) -> None:
        self.settings.tracking = bool(self.var_track.get())

    # ------------------------------------------------------------------
    # Ligar / gravar / foto
    # ------------------------------------------------------------------
    def toggle_camera(self) -> None:
        if not self._ui_running:
            if not self.pipeline.start():
                messagebox.showerror("Erro", f"Não foi possível abrir a fonte de vídeo:\n{self.cfg.source}")
                return
            self._set_running_ui(True)
        else:
            self._cancel_zone()
            self.pipeline.stop()
            self._set_running_ui(False)

    def _set_running_ui(self, running: bool) -> None:
        self._ui_running = running
        if running:
            self.btn_toggle.config(text="⏹ Desligar", bg=ORANGE)
            self.btn_record.config(state=tk.NORMAL)
            self.btn_snap.config(state=tk.NORMAL)
        else:
            self.btn_toggle.config(text="▶ Ligar", bg=GREEN)
            self.btn_record.config(state=tk.DISABLED)
            self.btn_snap.config(state=tk.DISABLED)
            self._set_recording_ui(False)
            self.canvas.delete("frame")
            self._img_item = None
            self._dw = self._dh = 0
            self._last_seq = -1
            self.lbl_count.config(text="Objetos: 0")
            self.lbl_fps.config(text="0 FPS · 0 ms")
            self.lbl_classes.config(text="")

    def _set_recording_ui(self, rec: bool) -> None:
        self._ui_recording = rec
        if rec:
            self.btn_record.config(text="⏹ Parar gravação", bg="#555555")
            self.lbl_rec.config(text="🔴 GRAVANDO")
        else:
            self.btn_record.config(text="⏺ Iniciar gravação", bg=RED)
            self.lbl_rec.config(text="")

    def toggle_recording(self) -> None:
        if not self.pipeline.recording:
            if self.pipeline.start_recording() is None:
                messagebox.showerror("Erro", "Não foi possível iniciar a gravação.")
                return
            self._set_recording_ui(True)
        else:
            self.pipeline.stop_recording()
            self._set_recording_ui(False)

    def take_snapshot(self) -> None:
        self.pipeline.snapshot()

    # ------------------------------------------------------------------
    # Zonas (desenho com o mouse)
    # ------------------------------------------------------------------
    def _toggle_zone_drawing(self) -> None:
        if not self._drawing:
            if self._dw == 0:
                messagebox.showinfo("Zonas", "Ligue a câmera primeiro para desenhar uma zona.")
                return
            self._drawing = True
            self._draft = []
            self.btn_zone.config(text="✔ Concluir zona", bg=GREEN)
            self.lbl_zone_info.config(
                text="Clique esquerdo: adiciona ponto · Direito/Concluir: fecha · Esc: cancela")
        else:
            self._finish_zone()

    def _on_canvas_left(self, event) -> None:
        if not self._drawing or self._dw == 0:
            return
        nx = (event.x - self._ox) / self._dw
        ny = (event.y - self._oy) / self._dh
        if 0 <= nx <= 1 and 0 <= ny <= 1:
            self._draft.append([nx, ny])
            self._draw_draft()

    def _finish_zone(self) -> None:
        if not self._drawing:
            return
        if len(self._draft) < 3:
            messagebox.showwarning("Zonas", "Uma zona precisa de pelo menos 3 pontos.")
            return
        name = simpledialog.askstring("Nova zona", "Nome da zona:",
                                      initialvalue=f"Zona {self.zones.count() + 1}",
                                      parent=self.root)
        if name is not None:
            final = self.zones.add(name, self._draft)
            self.zones.save(self.cfg.zones_file)
            self.events.log("zone_created", zone=final)
        self._cancel_zone()
        self._update_zone_info()

    def _cancel_zone(self) -> None:
        self._drawing = False
        self._draft = []
        self.canvas.delete("draft")
        self.btn_zone.config(text="✏ Desenhar zona", bg="#333333")
        self._update_zone_info()

    def _clear_zones(self) -> None:
        if self.zones.count() and messagebox.askyesno("Zonas", "Remover todas as zonas?"):
            self.zones.clear()
            self.zones.save(self.cfg.zones_file)
            self.events.log("zones_cleared")
            self._update_zone_info()

    def _update_zone_info(self) -> None:
        if not self._drawing:
            self.lbl_zone_info.config(text=f"Zonas ativas: {self.zones.count()}")

    def _draw_draft(self) -> None:
        self.canvas.delete("draft")
        pts = [(self._ox + x * self._dw, self._oy + y * self._dh) for x, y in self._draft]
        for x, y in pts:
            self.canvas.create_oval(x - 4, y - 4, x + 4, y + 4, fill="#ffd600",
                                    outline="", tags="draft")
        if len(pts) >= 2:
            flat = [c for p in pts for c in p]
            self.canvas.create_line(*flat, fill="#ffd600", width=2, tags="draft")
        self.canvas.tag_raise("draft")

    # ------------------------------------------------------------------
    # Atualização periódica
    # ------------------------------------------------------------------
    def _poll(self) -> None:
        if self._closing:
            return

        # Alertas vindos da thread de trabalho
        try:
            while True:
                msg = self.pipeline.event_queue.get_nowait()
                self.lst_alerts.insert(0, msg)
                if self.lst_alerts.size() > 50:
                    self.lst_alerts.delete(50, tk.END)
        except queue.Empty:
            pass

        # A thread terminou sozinha (fim do vídeo, erro...)
        if self._ui_running and not self.pipeline.running:
            self.pipeline.stop()
            self._set_running_ui(False)
        if self._ui_recording and not self.pipeline.recording:
            self._set_recording_ui(False)

        # Novo quadro
        res = self.pipeline.latest() if self._ui_running else None
        if res is not None and res.seq != self._last_seq:
            self._last_seq = res.seq
            self._render(res)

        # Status
        status = self.pipeline.status
        color = TEXT if status in ("Ativo", "Parado") else ORANGE
        if status == "SEM IMAGEM" or status.startswith("Erro"):
            color = "#ff5252"
        self.lbl_status.config(text=f"Status: {status}", fg=color)

        self.root.after(15, self._poll)

    def _render(self, res: FrameResult) -> None:
        frame = res.frame
        h, w = frame.shape[:2]
        cw = max(self.canvas.winfo_width(), 1)
        ch = max(self.canvas.winfo_height(), 1)
        scale = min(cw / w, ch / h)
        dw, dh = max(int(w * scale), 1), max(int(h * scale), 1)

        small = cv2.resize(frame, (dw, dh), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        self._photo = ImageTk.PhotoImage(Image.fromarray(rgb))

        self._dw, self._dh = dw, dh
        self._ox, self._oy = (cw - dw) // 2, (ch - dh) // 2
        if self._img_item is None:
            self._img_item = self.canvas.create_image(self._ox, self._oy, anchor="nw",
                                                      image=self._photo, tags="frame")
        else:
            self.canvas.coords(self._img_item, self._ox, self._oy)
            self.canvas.itemconfig(self._img_item, image=self._photo)
        self.canvas.tag_lower("frame")
        if self._drawing:
            self._draw_draft()

        self.lbl_count.config(text=f"Objetos: {res.n_objects}")
        self.lbl_fps.config(text=f"{res.fps:.0f} FPS · {res.infer_ms:.0f} ms")
        self.lbl_classes.config(
            text=" · ".join(f"{k}: {v}" for k, v in sorted(res.class_counts.items())))

    # ------------------------------------------------------------------
    def on_close(self) -> None:
        self._closing = True
        self.pipeline.stop()
        self.zones.save(self.cfg.zones_file)
        self.events.close()
        self.root.destroy()
