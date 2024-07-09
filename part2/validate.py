from ultralytics import YOLO

# Carregar o modelo YOLO
model = YOLO('models/mid_finger_yolov8n_764imgs.pt')

# Lista de fontes de vídeo
sources = [
    'D:\\Downloads\\Inferencia\\Flipping dogs - Original.mp4',
    'D:\\Downloads\\Inferencia\\Flipping the Bird - Supercut-Original.mp4',
    'D:\\Downloads\\Inferencia\\HowToFlip.mp4',
    'https://youtu.be/Io2vcwM5aaM'
]

# Loop para rodar o modelo para cada fonte
for source in sources:
    results = model.predict(source, stream=True, save=True, conf=0.5, show_conf=True, show_labels=True)
    for result in results:
        print(result.probs)