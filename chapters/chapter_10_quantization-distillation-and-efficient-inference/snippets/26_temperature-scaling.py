import copy
import torch
import torch.nn.functional as F
def self_distillation(model, train_loader, num_rounds=3):
    """
    Self-distillation: model teaches itself through iterative refinement
