def sobel_edge_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
     # Apply Sobel operators
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # Compute gradient magnitude
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    # Compute gradient direction
    direction = np.arctan2(sobel_y, sobel_x)
    return magnitude, direction, sobel_x, sobel_y
# Apply Sobel edge detection
mag, direction, grad_x, grad_y = sobel_edge_detection(image)
