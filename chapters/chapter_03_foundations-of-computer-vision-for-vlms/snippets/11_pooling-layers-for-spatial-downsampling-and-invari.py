class PoolingComparison(nn.Module):
    """Demonstrate different pooling operations"""

    def __init__(self):
        super(PoolingComparison, self).__init__()

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))  # Output size 7x7
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))    # Global pooling

    def forward(self, x):
        return {
            'max_pool': self.max_pool(x),
            'avg_pool': self.avg_pool(x),
            'adaptive_pool': self.adaptive_pool(x),
            'global_pool': self.global_pool(x).squeeze()  # Remove spatial dimensions
        }

# Demonstrate pooling operations
pooling_demo = PoolingComparison()
feature_maps = torch.randn(1, 64, 14, 14)  # 64 feature maps of size 14x14
pooling_results = pooling_demo(feature_maps)

print(f"Input feature maps: {feature_maps.shape}")
for pool_type, result in pooling_results.items():
    print(f"{pool_type}: {result.shape}")
