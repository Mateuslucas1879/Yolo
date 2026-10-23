import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO


class YOLOApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.window.geometry("900x650")
        self.window.config(bg="#1e1e1e")

        # 1. Carrega o modelo YOLO
        self.model = YOLO("recordings/yolo11n.pt")

        # Variáveis de controle da webcam
        self.cap = None
        self.is_running = False

        # --- LAYOUT DA INTERFACE ---
        # Título principal
        title_label = tk.Label(
            window,
            text="Detector YOLOv11 - Desktop",
            font=("Arial", 16, "bold"),
            bg="#1e1e1e",
            fg="white"
        )
        title_label.pack(pady=10)

        # Painel principal (divide a tela em vídeo e controles)
        main_frame = tk.Frame(window, bg="#1e1e1e")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Frame para o Vídeo (Esquerda)
        self.video_label = tk.Label(main_frame, bg="black")
        self.video_label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Frame para os Controles (Direita)
        control_frame = tk.Frame(main_frame, bg="#2d2d2d", width=250)
        control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10)

        # Botão Ligar/Desligar Câmera
        self.btn_toggle = tk.Button(
            control_frame,
            text="Ligar Câmera",
            font=("Arial", 11, "bold"),
            bg="#4CAF50",
            fg="white",
            command=self.toggle_camera,
            cursor="hand2"
        )
        self.btn_toggle.pack(fill=tk.X, padx=10, pady=20)

        # Rótulo e Slider para Confiança
        lbl_conf = tk.Label(
            control_frame,
            text="Confiança Mínima:",
            font=("Arial", 10),
            bg="#2d2d2d",
            fg="white"
        )
        lbl_conf.pack(anchor="w", padx=10)

        self.slider_conf = ttk.Scale(
            control_frame,
            from_=0.1,
            to=1.0,
            value=0.5,
            orient=tk.HORIZONTAL
        )
        self.slider_conf.pack(fill=tk.X, padx=10, pady=5)

        # Botão Sair
        btn_exit = tk.Button(
            control_frame,
            text="Sair",
            font=("Arial", 11, "bold"),
            bg="#f44336",
            fg="white",
            command=self.on_closing,
            cursor="hand2"
        )
        btn_exit.pack(fill=tk.X, padx=10, pady=20, side=tk.BOTTOM)

        # Protocolo para fechar a janela corretamente no 'X'
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)

    def toggle_camera(self):
        if not self.is_running:
            # Tenta abrir a webcam
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                print("Erro ao acessar a webcam.")
                return

            self.is_running = True
            self.btn_toggle.config(text="Desligar Câmera", bg="#ff9800")
            self.update_frame()
        else:
            self.is_running = False
            if self.cap:
                self.cap.release()
            self.btn_toggle.config(text="Ligar Câmera", bg="#4CAF50")
            self.video_label.config(image="")

    def update_frame(self):
        if self.is_running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # Pega o valor atual do slider de confiança
                conf_val = self.slider_conf.get()

                # Processa com o YOLO
                results = self.model(frame, conf=conf_val, stream=True)
                for r in results:
                    annotated_frame = r.plot()

                # Converte o frame do OpenCV (BGR) para o formato do Pillow (RGB)
                cv2image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(cv2image)

                # Redimensiona mantendo a proporção para caber na interface (opcional)
                img = img.resize((640, 480))

                imgtk = ImageTk.PhotoImage(image=img)

                # Atualiza a imagem na tela
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)

            # Chama a função novamente após 10 milissegundos para manter o fluxo
            self.window.after(10, self.update_frame)

    def on_closing(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
        self.window.destroy()


# Execução da Aplicação
if __name__ == "__main__":
    root = tk.Tk()
    app = YOLOApp(root, "Aplicação Desktop - YOLOv11")
    root.mainloop()