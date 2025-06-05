from ultralytics import YOLO

# Load a model
model = YOLO("yolo11s.pt")  # load an official model

# Export the model
model.export(format="onnx")
