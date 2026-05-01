def extract_orb_features(image, n_features=500):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  # Create ORB detector (patent-free alternative)
 orb = cv2.ORB_create(nfeatures=n_features)
 # Detect keypoints and compute descriptors
 keypoints, descriptors = orb.detectAndCompute(gray, None)
 return keypoints, descriptors
# ORB provides similar functionality to SURF but is patent-free
keypoints, descriptors = extract_orb_features(image)
print(f"ORB descriptors shape: {descriptors.shape}")  # (n_keypoints, 32)
