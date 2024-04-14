from ultralytics import YOLO

def yolov8_training(yaml_path):
    # Load a model
    model = YOLO("yolov8n.yaml")  # build a new model from scratch

    if __name__ == '__main__':
            # Use the model
        results = model.train(data=yaml_path, epochs=100, workers=2)  # train the model    # put in path location

