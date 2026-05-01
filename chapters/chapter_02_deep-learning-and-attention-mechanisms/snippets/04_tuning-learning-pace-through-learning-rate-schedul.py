Var(W) =
He initialization: Used for ReLU-based activations:
Var(W) =
Most transformer layers use small standard deviations (e.g., 0.02) and zero biases.
When fine-tuning VLMs, it is common to use pretrained vision encoders (e.g., ViT) or language models (e.g., BERT), freezing parts of the model and initializing cross-attention heads. Transformers use layer normalization instead of BatchNorm to stabilize training and prevent gradient collapse. Given an input vector , LayerNorm standardizes the values across the feature dimension as follows:
LayerNorm(x)   =
Where:
μ is the mean across features,
 is the variance,
