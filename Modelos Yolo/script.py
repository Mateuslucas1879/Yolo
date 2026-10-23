import os
import time
from datetime import datetime
import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO

class YOLOApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.window.geometry("1050x680")
        self.window.config(bg="#121212")

        # 1. Modelo YOLO
        self.model = YOLO("recordings/yolo11n.pt")

        # Variáveis de Controle
        self.cap = None
        self.is_running = False
        self.is_recording = False
        self.video_writer = None
        self.out_filename = ""

        # Variáveis para FPS
        self.prev_time = 0
        self.fps = 0

        # Cores do Tema
        self.COLOR_BG = "#121212"
        self.COLOR_PANEL = "#1e1e1e"
        self.COLOR_ACCENT = "#00adb5"
        self.COLOR_TEXT = "#eeeeee"
        self.COLOR_GREEN = "#2e7d32"
        self.COLOR_RED = "#c62828"
        self.COLOR_ORANGE = "#ef6c00"

        # Estilo do Tkinter (para sliders e barras)
        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.style.configure(
            "TScale",
            background=self.COLOR_PANEL,
            troughcolor="#333333",
            sliderlength=15
        )

        # --- CONSTRUÇÃO DA INTERFACE ---
        self.create_widgets()

        # Protocolo de fechamento
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)

    def create_widgets(self):
        # Header / Cabeçalho
        header_frame = tk.Frame(self.window, bg=self.COLOR_PANEL, height=50)
        header_frame.pack(fill=tk.X, side=tk.TOP)

        title_label = tk.Label(
            header_frame,
            text="🔍 YOLOv11 Studio - Visão Computacional Desktop",
            font=("Segoe UI", 14, "bold"),
            bg=self.COLOR_PANEL,
            fg=self.COLOR_ACCENT
        )
        title_label.pack(side=tk.LEFT, padx=20, pady=10)

        self.rec_status_label = tk.Label(
            header_frame,
            text="",
            font=("Segoe UI", 11, "bold"),
            bg=self.COLOR_PANEL,
            fg=self.COLOR_RED
        )
        self.rec_status_label.pack(side=tk.RIGHT, padx=20)

        # Container Principal
        main_container = tk.Frame(self.window, bg=self.COLOR_BG)
        main_container.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

        # --- ÁREA DO VÍDEO (ESQUERDA) ---
        video_frame = tk.Frame(main_container, bg="black", bd=2, relief=tk.SOLID)
        video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.video_label = tk.Label(video_frame, bg="black")
        self.video_label.pack(fill=tk.BOTH, expand=True)

        # --- PAINEL DE CONTROLES (DIREITA) ---
        control_frame = tk.Frame(main_container, bg=self.COLOR_PANEL, width=280)
        control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(15, 0))
        control_frame.pack_propagate(False)

        # Seção 1: Controles da Câmera
        lbl_sec_cam = tk.Label(
            control_frame, text="CONTROLES", font=("Segoe UI", 10, "bold"),
            bg=self.COLOR_PANEL, fg=self.COLOR_ACCENT
        )
        lbl_sec_cam.pack(anchor="w", padx=15, pady=(15, 5))

        self.btn_toggle = tk.Button(
            control_frame, text="▶ Ligar Câmera", font=("Segoe UI", 10, "bold"),
            bg=self.COLOR_GREEN, fg="white", activebackground="#1b5e20",
            activeforeground="white", relief=tk.FLAT, cursor="hand2", command=self.toggle_camera
        )
        self.btn_toggle.pack(fill=tk.X, padx=15, pady=5)

        self.btn_record = tk.Button(
            control_frame, text="⏺ Iniciar Gravação", font=("Segoe UI", 10, "bold"),
            bg=self.COLOR_RED, fg="white", activebackground="#b71c1c",
            activeforeground="white", relief=tk.FLAT, cursor="hand2",
            command=self.toggle_recording, state=tk.DISABLED
        )
        self.btn_record.pack(fill=tk.X, padx=15, pady=5)

        self.btn_snapshot = tk.Button(
            control_frame, text="📸 Capturar Foto", font=("Segoe UI", 10, "bold"),
            bg="#333333", fg=self.COLOR_TEXT, activebackground="#444444",
            activeforeground="white", relief=tk.FLAT, cursor="hand2",
            command=self.take_snapshot, state=tk.DISABLED
        )
        self.btn_snapshot.pack(fill=tk.X, padx=15, pady=5)

        ttk.Separator(control_frame, orient='horizontal').pack(fill=tk.X, padx=15, pady=15)

        # Seção 2: Configurações do Modelo
        lbl_sec_conf = tk.Label(
            control_frame, text="AJUSTE DE PARÂMETROS", font=("Segoe UI", 10, "bold"),
            bg=self.COLOR_PANEL, fg=self.COLOR_ACCENT
        )
        lbl_sec_conf.pack(anchor="w", padx=15, pady=(0, 5))

        self.lbl_conf_val = tk.Label(
            control_frame, text="Confiança Mínima: 50%", font=("Segoe UI", 9),
            bg=self.COLOR_PANEL, fg=self.COLOR_TEXT
        )
        self.lbl_conf_val.pack(anchor="w", padx=15)

        self.slider_conf = ttk.Scale(
            control_frame, from_=0.1, to=1.0, value=0.5,
            orient=tk.HORIZONTAL, command=self.update_conf_label
        )
        self.slider_conf.pack(fill=tk.X, padx=15, pady=5)

        ttk.Separator(control_frame, orient='horizontal').pack(fill=tk.X, padx=15, pady=15)

        # Seção 3: Estatísticas em Tempo Real (Cards)
        lbl_sec_stats = tk.Label(
            control_frame, text="ESTATÍSTICAS", font=("Segoe UI", 10, "bold"),
            bg=self.COLOR_PANEL, fg=self.COLOR_ACCENT
        )
        lbl_sec_stats.pack(anchor="w", padx=15, pady=(0, 5))

        # Card Objetos
        card_obj = tk.Frame(control_frame, bg="#2a2a2a", bd=1, relief=tk.SOLID)
        card_obj.pack(fill=tk.X, padx=15, pady=5)

        tk.Label(card_obj, text="Objetos Detectados", font=("Segoe UI", 8), bg="#2a2a2a", fg="#aaa").pack(anchor="w", padx=8, pady=(4, 0))
        self.lbl_count = tk.Label(card_obj, text="0", font=("Segoe UI", 14, "bold"), bg="#2a2a2a", fg=self.COLOR_TEXT)
        self.lbl_count.pack(anchor="w", padx=8, pady=(0, 4))

        # Card FPS
        card_fps = tk.Frame(control_frame, bg="#2a2a2a", bd=1, relief=tk.SOLID)
        card_fps.pack(fill=tk.X, padx=15, pady=5)

        tk.Label(card_fps, text="Taxa de Quadros (FPS)", font=("Segoe UI", 8), bg="#2a2a2a", fg="#aaa").pack(anchor="w", padx=8, pady=(4, 0))
        self.lbl_fps = tk.Label(card_fps, text="0 FPS", font=("Segoe UI", 14, "bold"), bg="#2a2a2a", fg=self.COLOR_ACCENT)
        self.lbl_fps.pack(anchor="w", padx=8, pady=(0, 4))

        # Botão Sair
        btn_exit = tk.Button(
            control_frame, text="Sair", font=("Segoe UI", 10, "bold"),
            bg="#333333", fg="#ff5252", activebackground="#ff5252", activeforeground="white",
            relief=tk.FLAT, cursor="hand2", command=self.on_closing
        )
        btn_exit.pack(fill=tk.X, padx=15, pady=15, side=tk.BOTTOM)

    def update_conf_label(self, val):
        percentage = int(float(val) * 100)
        self.lbl_conf_val.config(text=f"Confiança Mínima: {percentage}%")

    def toggle_camera(self):
        if not self.is_running:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Erro", "Não foi possível acessar a webcam.")
                return

            self.is_running = True
            self.btn_toggle.config(text="⏹ Desligar Câmera", bg=self.COLOR_ORANGE)
            self.btn_record.config(state=tk.NORMAL)
            self.btn_snapshot.config(state=tk.NORMAL)
            self.update_frame()
        else:
            if self.is_recording:
                self.stop_recording()

            self.is_running = False
            if self.cap:
                self.cap.release()

            self.btn_toggle.config(text="▶ Ligar Câmera", bg=self.COLOR_GREEN)
            self.btn_record.config(state=tk.DISABLED)
            self.btn_snapshot.config(state=tk.DISABLED)
            self.video_label.config(image="")
            self.lbl_fps.config(text="0 FPS")
            self.lbl_count.config(text="0")

    def toggle_recording(self):
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        if not self.is_running or not self.cap:
            return

        # Define resolução e codec para gravação
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Cria pasta 'recordings' se não existir
        os.makedirs("recordings", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.out_filename = os.path.join("recordings", f"yolo_record_{timestamp}.mp4")

        # Codec mp4v
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(self.out_filename, fourcc, 20.0, (width, height))

        if self.video_writer.isOpened():
            self.is_recording = True
            self.btn_record.config(text="⏹ Parar Gravação", bg="#555555")
            self.rec_status_label.config(text="🔴 GRAVANDO")
        else:
            messagebox.showerror("Erro", "Falha ao iniciar o arquivo de gravação.")

    def stop_recording(self):
        if self.is_recording:
            self.is_recording = False
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None

            self.btn_record.config(text="⏺ Iniciar Gravação", bg=self.COLOR_RED)
            self.rec_status_label.config(text="")
            messagebox.showinfo("Gravação Salva", f"Vídeo salvo com sucesso em:\n{self.out_filename}")

    def take_snapshot(self):
        if hasattr(self, 'current_annotated_frame') and self.current_annotated_frame is not None:
            os.makedirs("snapshots", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join("snapshots", f"snapshot_{timestamp}.png")
            cv2.imwrite(filename, self.current_annotated_frame)
            messagebox.showinfo("Foto Capturada", f"Imagem salva em:\n{filename}")

    def update_frame(self):
        if self.is_running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # Cálculo de FPS
                curr_time = time.time()
                self.fps = 1 / (curr_time - self.prev_time) if self.prev_time > 0 else 0
                self.prev_time = curr_time

                # Detecção YOLO
                conf_val = self.slider_conf.get()
                results = self.model(frame, conf=conf_val, stream=True)

                annotated_frame = frame.copy()
                total_objects = 0

                for r in results:
                    annotated_frame = r.plot()
                    total_objects = len(r.boxes)

                # Salva frame atual para foto/gravação
                self.current_annotated_frame = annotated_frame

                # Grava frame se a gravação estiver ativa
                if self.is_recording and self.video_writer:
                    self.video_writer.write(annotated_frame)

                # Atualiza Interface (Estatísticas)
                self.lbl_fps.config(text=f"{int(self.fps)} FPS")
                self.lbl_count.config(text=str(total_objects))

                # Redimensiona para caber no Tkinter
                cv2image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(cv2image)
                img = img.resize((720, 540))

                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)

            # Próximo frame
            self.window.after(10, self.update_frame)

    def on_closing(self):
        if self.is_recording:
            self.stop_recording()
        self.is_running = False
        if self.cap:
            self.cap.release()
        self.window.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = YOLOApp(root, "YOLOv11 Desktop Studio")
    root.mainloop()