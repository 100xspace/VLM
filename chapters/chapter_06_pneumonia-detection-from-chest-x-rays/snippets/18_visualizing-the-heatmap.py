import matplotlib.pyplot as plt

def overlay_heatmap(original_img_path, heatmap):
    # Load original image
    img = cv2.imread(original_img_path)
    img = cv2.resize(img, (224, 224))

    # Resize heatmap to match image
    heatmap = cv2.resize(heatmap, (224, 224))

    # Convert to color map (Jet: Blue=Cold, Red=Hot)
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

    # Superimpose
    superimposed_img = np.uint8(superimposed_img)# Adjust opacity
    cv2.imwrite("gradcam_result.jpg", superimposed_img)
    return superimposed_img
