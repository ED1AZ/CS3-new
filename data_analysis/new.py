import roboflow as roboflow
import supervision as sv
import inference
import cv2 as cv
import onnxruntime as ort
print(ort.get_available_providers())
api_key = "biVnTCggzj3GiRiSl5YD" 
#rf = roboflow.Roboflow(api_key=api_key)
#model = rf.workspace("ed1az").project("current-dataset-czyp8").version(2).model

img = cv.imread("data_analysis/annotated_frames/test/images/013_MOV-0008_jpg.rf.a241f21bedcfed362b3f6626471318dc.jpg")
#results = model.predict(img, confidence=40, overlap=30)

model = inference.get_model(model_id="current-dataset-czyp8/2", api_key=api_key)

results = model.infer(img, confidence=0.5)[0]

detections = sv.Detections.from_inference(results)
bounding_box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()
annotated_image = bounding_box_annotator.annotate(scene=img.copy(), detections=detections)
cv.imshow("annotated_image", annotated_image)
