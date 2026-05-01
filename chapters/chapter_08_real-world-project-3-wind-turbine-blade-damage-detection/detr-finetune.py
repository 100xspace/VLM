BATCH_SIZE = 8
class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, image_directory_path: str, image_processor, train: bool = True):
        annotation_file_path = os.path.join(image_directory_path, "_annotations.coco.json")
        super(CocoDetection, self).__init__(image_directory_path, annotation_file_path)
