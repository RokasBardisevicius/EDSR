import torch
import torch.nn as nn
from config import config

# Residual Block used in the EDSR model
class ResidualBlock(nn.Module):
    def __init__(self, num_features):
        super(ResidualBlock, self).__init__()
        # Define a sequence of convolutional layers with ReLU activation
        self.block = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        )

    def forward(self, x):
        # Skip connection: adds the input to the output of the block
        return x + self.block(x)

# Enhanced Deep Super-Resolution (EDSR) model
class EDSR(nn.Module):
    def __init__(self, scale_factor=config.scale, num_channels=config.num_channels, num_res_blocks=config.num_res_blocks):
        super(EDSR, self).__init__()
        self.scale_factor = scale_factor
        num_features = config.num_features  # Number of features in the residual blocks

        # Head: initial convolution layer that increases the feature depth
        self.head = nn.Conv2d(num_channels, num_features, kernel_size=3, padding=1)

        # Body: sequence of residual blocks
        self.body = nn.Sequential(
            *[ResidualBlock(num_features) for _ in range(num_res_blocks)]
        )

        # Tail: upscales the image back to the desired resolution
        self.tail = nn.Sequential(
            nn.Conv2d(num_features, num_features * (scale_factor ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(scale_factor),  # Upscales the image
            nn.Conv2d(num_features, num_channels, kernel_size=3, padding=1)  # Final convolution to adjust the number of channels
        )

    def forward(self, x):
        # Pass input through the head, body, and tail of the network
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        return x
