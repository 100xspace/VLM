student_soft = F.log_softmax(student_logits / temperature, dim=1)
 teacher_soft = F.softmax(teacher_logits / temperature, dim=1)
    # KL divergence between teacher and student
 distillation_loss = F.kl_div(
     student_soft,
     teacher_soft,
     reduction='batchmean'
 ) * (temperature ** 2)  # Scale by T^2 to balance magnitude
    # Hard targets: standard cross-entropy with ground truth
 hard_loss = F.cross_entropy(student_logits, labels)
    # Combined loss
 total_loss = alpha * distillation_loss + (1 - alpha) * hard_loss
    return total_loss, distillation_loss.item(), hard_loss.item()
# Example: Compare temperature effects
teacher_logits = torch.tensor([[10.0, 1.0, 0.5]])  # Teacher confident about class 0
student_logits = torch.tensor([[5.0, 3.0, 2.0]])   # Student less confident
print("Temperature effects on soft targets:")
for T in [1.0, 2.0, 5.0, 10.0]:
 teacher_probs = F.softmax(teacher_logits / T, dim=1)
 print(f"T={T:4.1f}: {teacher_probs.numpy()[0]}")
