from peft.tuners.lora import LoraLayer
class DoRALayer(LoraLayer):
    def forward(self, x):
        # Decompose weight magnitude + direction
    weight_norm = self.weight.norm(dim=0, keepdim=True)
    normalized_weight = self.weight / weight_norm
        return torch.matmul(x, normalized_weight.T) * weight_norm
# DoRA is often framework-integrated; logic shown explicitly
