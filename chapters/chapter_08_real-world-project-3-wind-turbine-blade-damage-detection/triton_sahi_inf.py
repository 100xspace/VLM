def callback(image: np.ndarray) -> sv.Detections:
    try:
        if not hasattr(model_local, "model"):
            model_local.model = YOLO("http://localhost:8000/yolo", task="detect")
        result = model_local.model(image, imgsz=640)[0]
        return sv.Detections.from_ultralytics(result)
    except Exception as e:
        raise Exception(f"Error occured durin inference: {e}")
# SAHI Slicer Configuration
slicer = sv.InferenceSlicer(
    callback=callback,
    slice_wh=(640, 640),
    overlap_wh=(160, 160), # Significant overlap to catch edge cases
    thread_workers=1,
)
