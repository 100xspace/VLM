from torchvision import transforms
from PIL import Image
def anyres_process(image: Image.Image, patch_size=224):
w, h = image.size
grids = []
    if h > w:
    grids = [(0, i * patch_size, w, (i+1) * patch_size)
                 for i in range(h // patch_size)]
    else:
    grids = [(i * patch_size, 0, (i+1) * patch_size, h)
                 for i in range(w // patch_size)]
crops = [image.crop(g) for g in grids]
    # Global thumbnail
global_view = image.resize((patch_size, patch_size))
    return [global_view] + crops
