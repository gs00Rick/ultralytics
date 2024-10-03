from ultralytics import YOLO



if __name__ == '__main__':
    # Load a model
    model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)

    # Train the model with 1 GPUs
    results = model.train(data="../datasets/coco8/coco8.yaml", epochs=100, imgsz=640, device=0, cache=True)










