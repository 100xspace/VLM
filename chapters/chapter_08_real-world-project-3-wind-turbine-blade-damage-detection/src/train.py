img_props = dataset_object.get_image_properties("train")
# Training Execution
model = YOLODetect(model="<yolo11x.pt|LOCAL_MODEL_PATH>").get_model()
trainer = YOLODetectTrainer(
    model=model,
    data=yaml_path,
    optimizer=optimizer,
    imgsz=img_props,
    device="0,1,2,3,4,5,6,7",  # Multi-GPU Support
)
trainer.train(data_path=yaml_path, epochs=200)
