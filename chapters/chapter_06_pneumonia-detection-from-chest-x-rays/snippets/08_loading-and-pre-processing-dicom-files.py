import pydicom
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
def dicom_to_grayscale_pil(dicom_path, window_center=-500, window_width=1500):
    """
    Convert DICOM to grayscale PIL Image with HU windowing.
    Args:
        dicom_path (str): Path to .dcm file.
        window_center (int): HU center for windowing (lungs: -500).
        window_width (int): HU width (lungs: 1500).
    Returns:
        PIL.Image: Grayscale image, normalized to [0, 255].
    """
    # 1. Load DICOM
    ds = pydicom.dcmread(dicom_path)
   # 2. Extract Pixel Array (handle different data types)
    if 'PixelData' not in ds:
        raise ValueError("No pixel data found in DICOM.")
    pixel_array = ds.pixel_array.astype(np.float32)
    # 3. Apply Modality LUT (HU conversion if CT; for X-ray, often identity)
    if ds.Modality == 'CT':
        # HU = pixel * slope + intercept (default: 1* + 0)
        pixel_array = pixel_array * ds.RescaleSlope + ds.RescaleIntercept
    # For X-ray (RX), skippixels are already intensity-scaled
    # 4. Clip to HU range and window (soft tissue focus)
    hu_min = window_center - window_width // 2
    hu_max = window_center + window_width // 2
    pixel_array = np.clip(pixel_array, hu_min, hu_max)
    # 5. Normalize to [0, 255] and invert if MONOCHROME1 (air=bright)
    pixel_array = (pixel_array - hu_min) / (hu_max - hu_min) * 255.0
    if ds.PhotometricInterpretation == 'MONOCHROME1':
        pixel_array = 255 - pixel_array  # Invert for standard view (air=black)

    # 6. Convert to uint8 and PIL
    pixel_array = pixel_array.astype(np.uint8)
    return Image.fromarray(pixel_array).convert('L')  # 'L' mode: grayscale
# Example Usage
dicom_path = "./rsna_data/stage_2_train_images/ID_0000.dcm"  # Replace with real path
gray_img = dicom_to_grayscale_pil(dicom_path)
# Visualize (Figure 6.2)
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
raw_img = pydicom.dcmread(dicom_path).pixel_array  # Raw for comparison
axes[0].imshow(raw_img, cmap='gray')
axes[0].set_title('Raw DICOM Pixels')
axes[1].imshow(gray_img, cmap='gray')
axes[1].set_title('Windowed Grayscale (Lung View)')
plt.tight_layout()
plt.savefig('dicom_windowing.png')  # Insert as Figure 6.2
plt.show()
In production pipelines, it is also important to preserve key metadata such as image orientation and pixel spacing, as these attributes can influence downstream resizing, spatial interpretation, and model consistency across datasets.
Refer to the following figure:

Figure 6.2: Effect of lung windowing on raw DICOM chest X-Ray data
