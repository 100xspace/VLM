from skimage.feature import hog
from skimage import exposure
def extract_hog_features(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

   # Extract HOG features
   features, hog_image = hog(gray,
                              orientations=9,
                              pixels_per_cell=(8, 8),
                              cells_per_block=(2, 2),
                              visualize=True,
                              block_norm='L2-Hys')
   # Rescale HOG image for better visualization
   hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
return features, hog_image_rescaled
# Extract HOG features
hog_features, hog_viz = extract_hog_features(image)
print(f"HOG feature vector length: {len(hog_features)}")
