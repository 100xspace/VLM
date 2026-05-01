import torch
import torch.nn.functional as F
import cv2
import numpy as np

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_image, class_idx=None):
        # 1. Forward Pass
        output = self.model(input_image)

        if class_idx is None:
            class_idx = torch.argmax(output)

        # 2. Select score for backward pass
       self.model.zero_grad()       if output.shape[1] == 1 or output. dim() == 1: # binary single-logit case                score= output.view(-1)[0]       else: # multi-class case              if class_idx is None:                   class_idx = torch.argmax(output, dim=1).item()              score= output[:, class_idx]      score.backward()
        # 3. Get Gradients and Activations
        gradients = self.gradients[0]   # [Channels, Height, Width]
        activations = self.activations[0]

        # 4. Global Average Pooling of Gradients (Calculate Weights)
        weights = torch.mean(gradients, dim=(1, 2))

        # 5. Weighted Combination of Activations
        heatmap = torch.zeros_like(activations[0])
        for i, w in enumerate(weights):
            heatmap += w * activations[i]

        # 6. Apply ReLU and Normalize
        heatmap = F.relu(heatmap)
        heatmap = heatmap / torch.max(heatmap)

        return heatmap.detach().cpu().numpy()

# Usage Example
# target_layer = model.densenet.features.denseblock4.denselayer16
# cam = GradCAM(model, target_layer)
# heatmap = cam.generate_heatmap(input_tensor)
