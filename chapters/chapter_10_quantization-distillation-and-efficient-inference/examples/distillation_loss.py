"""Knowledge distillation losses for compact VLM students."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class DistillationWeights:
    hard_label: float = 0.4
    soft_logits: float = 0.5
    feature: float = 0.1


def soft_target_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float = 3.0) -> torch.Tensor:
    """KL loss between softened teacher and student distributions."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature**2)


def feature_alignment_loss(student_features: torch.Tensor, teacher_features: torch.Tensor) -> torch.Tensor:
    """Cosine-style feature matching for vision or multimodal embeddings."""
    student_features = F.normalize(student_features.float(), dim=-1)
    teacher_features = F.normalize(teacher_features.float(), dim=-1)
    return 1.0 - (student_features * teacher_features).sum(dim=-1).mean()


def vlm_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    student_features: torch.Tensor | None = None,
    teacher_features: torch.Tensor | None = None,
    temperature: float = 3.0,
    weights: DistillationWeights = DistillationWeights(),
) -> tuple[torch.Tensor, dict[str, float]]:
    """Combine supervised labels, softened logits, and optional feature matching."""
    hard = F.cross_entropy(student_logits, labels)
    soft = soft_target_loss(student_logits, teacher_logits, temperature=temperature)
    feature = student_logits.new_tensor(0.0)
    if student_features is not None and teacher_features is not None:
        feature = feature_alignment_loss(student_features, teacher_features)

    total = weights.hard_label * hard + weights.soft_logits * soft + weights.feature * feature
    metrics = {
        "total": float(total.detach()),
        "hard_label": float(hard.detach()),
        "soft_logits": float(soft.detach()),
        "feature": float(feature.detach()),
    }
    return total, metrics


if __name__ == "__main__":
    torch.manual_seed(7)
    student_logits = torch.randn(8, 12)
    teacher_logits = student_logits + 0.8 * torch.randn(8, 12)
    labels = torch.randint(0, 12, (8,))
    student_features = torch.randn(8, 256)
    teacher_features = student_features + 0.2 * torch.randn(8, 256)

    loss, metrics = vlm_distillation_loss(
        student_logits,
        teacher_logits,
        labels,
        student_features=student_features,
        teacher_features=teacher_features,
    )
    loss.backward()
    print(metrics)
