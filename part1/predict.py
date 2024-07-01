from ultralytics import YOLO

model = YOLO("yolov8n.pt")

# Define path to video file
source = "https://youtu.be/yn-TfAzobDI"

results = model.predict(source,stream=True, save=True, conf=0.5, classes=36,show_conf=True,show_labels=True)
for result in results:
    print(result.probs)