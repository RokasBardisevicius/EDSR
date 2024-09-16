import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F 

# Perceptual Loss using VGG19 features to compare high-level perceptual similarity
class PerceptualLoss(nn.Module):
    def __init__(self, feature_layer=9, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super(PerceptualLoss, self).__init__()
        # Load the VGG19 model and extract features up to the specified layer
        vgg = models.vgg19(weights='VGG19_Weights.IMAGENET1K_V1').features
        self.vgg = nn.Sequential(*list(vgg.children())[:feature_layer]).eval().to(device)
        # Freeze VGG model parameters
        for param in self.vgg.parameters():
            param.requires_grad = False
        self.criterion = nn.L1Loss()

    def forward(self, sr, hr):
        # Ensure that the input and VGG model are on the same device
        sr = sr.to(self.vgg[0].weight.device)
        hr = hr.to(self.vgg[0].weight.device)
        # Extract features from the super-resolved and high-resolution images
        sr_features = self.vgg(sr)
        hr_features = self.vgg(hr)
        # Calculate L1 loss between the feature maps
        return self.criterion(sr_features, hr_features)

# Edge-Aware Loss to penalize differences in edge structures between predicted and target images
class EdgeAwareLoss(nn.Module):
    def __init__(self):
        super(EdgeAwareLoss, self).__init__()
        # Sobel kernels for edge detection in x and y directions
        sobel_kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.sobel_x = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.sobel_y = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.sobel_x.weight.data = sobel_kernel_x
        self.sobel_y.weight.data = sobel_kernel_y

    def forward(self, pred, target):
        # Ensure that the filters and inputs are on the same device
        device = pred.device
        self.sobel_x = self.sobel_x.to(device)
        self.sobel_y = self.sobel_y.to(device)

        pred_edge = []
        target_edge = []
        # Iterate over each channel to compute edge maps
        for c in range(pred.shape[1]):
            # Compute edge maps for predicted and target images
            pred_edge_x = F.conv2d(pred[:, c:c+1, :, :], self.sobel_x.weight)
            pred_edge_y = F.conv2d(pred[:, c:c+1, :, :], self.sobel_y.weight)
            target_edge_x = F.conv2d(target[:, c:c+1, :, :], self.sobel_x.weight)
            target_edge_y = F.conv2d(target[:, c:c+1, :, :], self.sobel_y.weight)

            # Compute magnitude of the gradient
            pred_edge.append(torch.sqrt(pred_edge_x ** 2 + pred_edge_y ** 2))
            target_edge.append(torch.sqrt(target_edge_x ** 2 + target_edge_y ** 2))

        # Concatenate edges for all channels
        pred_edge = torch.cat(pred_edge, dim=1)
        target_edge = torch.cat(target_edge, dim=1)

        # Calculate L1 loss between predicted and target edge maps
        return F.l1_loss(pred_edge, target_edge)

# Combined Loss that includes L1, Perceptual, and Edge-Aware Loss
class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.l1_loss = nn.L1Loss()  # L1 loss between predicted and target images
        self.perceptual_loss = PerceptualLoss()  # Perceptual loss using VGG19 features
        self.edge_aware_loss = EdgeAwareLoss()  # Edge-aware loss for edge structure preservation

    def forward(self, sr, hr):
        # Resize high-resolution image to match the super-resolved image's size
        hr_resized = F.interpolate(hr, size=sr.shape[-2:], mode='bilinear', align_corners=False)

        # Compute individual losses
        l1 = self.l1_loss(sr, hr_resized)
        perceptual = self.perceptual_loss(sr, hr_resized)
        edge_loss = self.edge_aware_loss(sr, hr_resized)

        # Combine losses with weights
        return l1 + 0.01 * perceptual + 0.01 * edge_loss
