import torch
import torch.nn as nn
import torch.nn.functional as F
def distillation_loss(student_logits, teacher_logits, labels, temperature=3.0, alpha=0.7):
 """
