import cv2
import numpy as np

def preprocess_image(image):
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
     # Gaussian blur for noise reduction
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Histogram equalization for contrast enhancement
    equalized = cv2.equalizeHist(blurred)
    # Normalization to [0, 1] range
    normalized = equalized.astype(np.float32) / 255.0
    return normalized
