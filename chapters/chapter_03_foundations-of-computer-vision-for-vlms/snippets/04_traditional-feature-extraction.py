# SIFT implementation example
import cv2
def extract_sift_features(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
     # Create SIFT detector
    sift = cv2.SIFT_create()
     # Detect keypoints and compute descriptors
    keypoints, descriptors = sift.detectAndCompute(gray, None)
     # Draw keypoints for visualization
    img_keypoints = cv2.drawKeypoints(image, keypoints, None,
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return keypoints, descriptors, img_keypoints
# Usage
keypoints, descriptors, visualized = extract_sift_features(image)
print(f"Number of keypoints: {len(keypoints)}")
print(f"Descriptor shape: {descriptors.shape}")  # (n_keypoints, 128)
