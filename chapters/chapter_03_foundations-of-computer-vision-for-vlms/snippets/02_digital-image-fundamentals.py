# Example: Loading and examining RGB image structure
import cv2
import numpy as np
# Load an RGB image
image = cv2.imread('sample_image.jpg')
print(f"Image shape: {image.shape}")  # (height, width, channels)
print(f"Data type: {image.dtype}")    # typically uint8
# Access individual channels
blue_channel = image[:, :, 0]
green_channel = image[:, :, 1]
red_channel = image[:, :, 2]
