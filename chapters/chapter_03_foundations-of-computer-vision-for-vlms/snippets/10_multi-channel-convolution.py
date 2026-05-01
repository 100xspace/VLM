class BasicConvBlock(nn.Module):
    """Basic convolution block with BatchNorm and ReLU"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(BasicConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Standard order: Conv -> BatchNorm -> ReLU
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
# Example: RGB input to feature maps
conv_block = BasicConvBlock(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)
input_rgb = torch.randn(4, 3, 224, 224)  # Batch of 4 RGB images
output_features = conv_block(input_rgb)
print(f"RGB Input: {input_rgb.shape}")
print(f"Feature Maps: {output_features.shape}")
