import os
from PIL import Image
import torch
from torch.utils.data import Dataset

class PneumoniaDataset(Dataset):
    def __init__(self, dataframe, root_dir, transform=None):
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.dataframe.iloc[idx]["image_id"])
        try:
            image = Image.open(img_name).convert('RGB')
        except:
            image = Image.new('RGB', (224, 224))  # fallback for bad files

        label = torch.tensor(self.dataframe.iloc[idx]["label"], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)
        return image, label
