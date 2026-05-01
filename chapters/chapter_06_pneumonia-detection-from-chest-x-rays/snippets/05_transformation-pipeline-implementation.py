import torch
from torchvision import transforms
# ImageNet statistics
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

def get_transforms(phase='train'):
    if phase == 'train':
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5), # Safe for Pneumonia
            transforms.RandomRotation(10),          # Small rotation only
            transforms.ToTensor(),                  # Converts to [0,1] range
            transforms.Normalize(mean=MEAN, std=STD)
        ])
    else:
        # Validation/Test: No augmentation, just resizing and norm
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD)
        ])
