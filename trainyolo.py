if __name__ == '__main__':
    from ultralytics import YOLO

    # Load a pre-trained YOLO model
    model = YOLO("./runs/detect/train3/weights/best.pt")
    model.info()

    # Train the model on a custom dataset
    # results = model.train(data='externals/yolov9/PlastOPol-1/data.yaml', epochs=30, imgsz=640, batch=-1)
    amp = False
    results = model.train(
        data='PlastOPol-1/data.yaml',
        epochs=100, 
        imgsz=640, 
        batch=-1
        )