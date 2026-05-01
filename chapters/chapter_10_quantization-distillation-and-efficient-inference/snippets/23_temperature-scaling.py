import torch
import torch.nn as nn
import torch.nn.functional as F

class VLMDistillationWrapper(nn.Module):
    """Wrapper for distilling a Vision-Language Model with multi-level supervision."""
    def __init__(self, teacher_model, student_model):
        super().__init__(); self.teacher, self.student = teacher_model, student_model
        for p in self.teacher.parameters(): p.requires_grad = False
        self.teacher.eval()
        self.vision_projector = nn.Linear(student_model.vision_encoder.output_dim, teacher_model.vision_encoder.output_dim)
        self.language_projector = nn.Linear(student_model.language_model.hidden_size, teacher_model.language_model.hidden_size)

    def forward(self, images, input_ids, attention_mask):
        """Compute multi-level distillation losses."""
        with torch.no_grad():
            t_out = self.teacher(images=images, input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            t_v, t_l, t_logits = t_out.vision_features, t_out.language_hidden_states[-1], t_out.logits

        s_out = self.student(images=images, input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        s_v, s_l, s_logits = s_out.vision_features, s_out.language_hidden_states[-1], s_out.logits
        s_vp, s_lp = self.vision_projector(s_v), self.language_projector(s_l)

        losses = {
            "vision": F.mse_loss(s_vp, t_v),
            "language": F.mse_loss(s_lp, t_l)
        }
        T = 3.0
        losses["output"] = F.kl_div(F.log_softmax(s_logits / T, dim=-1), F.softmax(t_logits / T, dim=-1), reduction="batchmean") * (T ** 2)

        if hasattr(s_out, "attentions") and hasattr(t_out, "attentions"):
            losses["attention"] = self._attention_distillation_loss(s_out.attentions, t_out.attentions)
        return losses

    def _attention_distillation_loss(self, student_attns, teacher_attns):
        """Match attention distributions between student and teacher."""
        loss = 0.0; n = min(len(student_attns), len(teacher_attns))
        for s_attn, t_attn in zip(student_attns[-n:], teacher_attns[-n:]):
            s_avg, t_avg = s_attn.mean(dim=1), t_attn.mean(dim=1)
            loss += F.mse_loss(s_avg, t_avg)
        return loss / max(n, 1)

def train_distilled_vlm(teacher, student, train_loader, num_epochs=10, device="cuda"):
    """Train a student Vision-Language Model through knowledge distillation."""
    wrapper = VLMDistillationWrapper(teacher, student).to(device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-4)
    w = {"vision": 1.0, "language": 1.0, "output": 2.0, "attention": 0.5}

    for epoch in range(num_epochs):
        wrapper.train(); total_loss = 0.0
        for batch_idx, batch in enumerate(train_loader):
            images = batch["images"].to(device); input_ids = batch["input_ids"].to(device); attention_mask = batch["attention_mask"].to(device)
            losses = wrapper(images, input_ids, attention_mask)
            combined_loss = sum(w.get(k, 1.0) * v for k, v in losses.items())

            optimizer.zero_grad(); combined_loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step(); total_loss += combined_loss.item()

            if batch_idx % 50 == 0:
                loss_str = ", ".join([f"{k}: {v.item():.4f}" for k, v in losses.items()])
                print(f"Epoch {epoch}, Batch {batch_idx}, {loss_str}")

        avg_loss = total_loss / max(len(train_loader), 1)
        print(f"Epoch {epoch} completed. Average loss: {avg_loss:.4f}")
    return student
