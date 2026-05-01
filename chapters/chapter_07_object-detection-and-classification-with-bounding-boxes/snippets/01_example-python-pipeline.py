from ultralytics import YOLO
import cv2

# 1. Load a YOLO model (YOLOv8n as an example)
model = YOLO("yolov8n.pt")  # pre-trained weights

# 2. Train on a custom dataset
results = model.train(
    data="pneumonia_det.yaml",  # dataset YAML file
    imgsz=640,
    epochs=100,
    batch=16,
    lr0=0.01,
    optimizer="SGD"
)

# 3. Evaluate the trained model on the validation split
metrics = model.val()
print(metrics.box.map50)      # mAP at IoU 0.50
print(metrics.box.map50_95)   # mAP at IoU 0.50–0.95

# 4. Run inference on a new image
pred = model("sample_image.png")[0]

# 5. Visualize with OpenCV
image = cv2.imread("sample_image.png")

for box in pred.boxes:
    x1, y1, x2, y2 = box.xyxy[0].int().tolist()
    cls_id = int(box.cls[0])
    score = float(box.conf[0])
    label = f"{model.names[cls_id]} {score:.2f}"

    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(image, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

cv2.imwrite("sample_detection.png", image)
