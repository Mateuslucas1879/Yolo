import cv2
from ultralytics import YOLO

# 1. Carrega o modelo pré-treinado (usaremos a versão nano 'n', que é super rápida para webcam)
model = YOLO("yolo11n.pt")

# 2. Inicializa a webcam (0 geralmente é a webcam padrão do computador)
cap = cv2.VideoCapture(0)

# Verifica se a webcam abriu corretamente
if not cap.isOpened():
    print("Erro ao acessar a webcam.")
    exit()

print("Pressione 'q' no teclado para sair.")

while cap.isOpened():
    # Lê um quadro (frame) da webcam
    success, frame = cap.read()

    if success:
        # 3. Realiza a inferência no frame atual
        # stream=True otimiza o uso de memória para vídeos contínuos
        results = model(frame, stream=True)

        # 4. Processa os resultados e desenha as caixas na imagem
        for r in results:
            annotated_frame = r.plot()  # Desenha as bounding boxes automaticamente

        # 5. Exibe o vídeo processado em uma janela
        cv2.imshow("YOLOv11 - Deteccao em Tempo Real", annotated_frame)

        # Sai do loop se a tecla 'q' for pressionada
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Libera a webcam e fecha as janelas abertas
cap.release()
cv2.destroyAllWindows()
