from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.pt')  # load an official model
# load a custom trained
model = YOLO('C:/Users/Administrator/Desktop/ObjectDetection/objectDetectionCode/ultralytics-main/objectDetection/runs/train/exp6/weights/best.pt')

# Export the model
model.export(format='onnx')
