def yolo2coco_bbox(bbox, img_w, img_h):
    x_center, y_center, w, h = bbox
    x_center *= img_w
    y_center *= img_h
    w *= img_w
    h *= img_h
    x_min = x_center - w / 2
    y_min = y_center - h / 2
    return [x_min, y_min, w, h]
