from ultralytics import YOLO

# Carrega um modelo pré-treinado leve (versão nano)
model = YOLO("yolo11n.pt")

# Realiza a inferência em uma imagem de exemplo
results = model("https://ultralytics.com/images/bus.jpg")

# Exibe o resultado com as caixas delimitadoras desenhadas
results[0].show()