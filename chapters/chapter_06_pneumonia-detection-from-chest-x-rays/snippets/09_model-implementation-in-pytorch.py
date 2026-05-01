import torch
import torch.nn as nn
from torchvision import models

class PneumoniaDenseNet(nn.Module):
    def __init__(self):
        super(PneumoniaDenseNet, self).__init__()

        # 1. Load Pre-trained Model
        self.densenet = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)

        # 2. Freeze Early Layers (Optional but recommended for small datasets)
        # This prevents the massive gradients from destroying pre-trained features
        for param in self.densenet.features.parameters():
            param.requires_grad = False

        # 3. Modify the Classifier Head
        # DenseNet121's classifier input features is usually 1024
        num_features = self.densenet.classifier.in_features

        self.densenet.classifier = nn.Sequential(
            nn.Linear(num_features, 512),  # Reduce dimension
            nn.ReLU(),                     # Activation
            nn.Dropout(0.3),               # Regularization to prevent overfitting
            nn.Linear(512, 1)              # Final Output: 1 Logit
        )

    def forward(self, x):
        return self.densenet(x)

# Instantiate
model = PneumoniaDenseNet()
