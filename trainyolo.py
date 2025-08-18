if __name__ == '__main__':
    from ultralytics import YOLO

    # Load a pre-trained YOLO model
    model = YOLO("yolov9s.yaml")
    model.info()

    # Train the model on a custom dataset
    # results = model.train(data='externals/yolov9/PlastOPol-1/data.yaml', epochs=30, imgsz=640, batch=-1)
    amp = False
    results = model.train(
        data='Litter-Detection-6/data.yaml', 
        epochs=300, 
        imgsz=640, 
        batch=-1
        )