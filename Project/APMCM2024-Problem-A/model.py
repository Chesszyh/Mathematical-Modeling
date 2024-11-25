import torch
import torch.nn as nn
from torchvision.models import resnet18, vgg16
from torch.nn import functional as F
import numpy as np

# 任务4：深度学习模型的定义和训练
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class UNetEnhancer(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        
        # 初始卷积层
        self.inc = DoubleConv(in_channels, 64)
        
        # 编码器路径
        resnet = resnet18(pretrained=True)
        self.encoder1 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )  # 输出尺寸减半
        self.encoder2 = resnet.layer1  # 尺寸不变
        self.encoder3 = resnet.layer2  # 尺寸减半
        self.encoder4 = resnet.layer3  # 尺寸减半
        
        # SE blocks
        self.se1 = SEBlock(64)
        self.se2 = SEBlock(64)
        self.se3 = SEBlock(128)
        self.se4 = SEBlock(256)
        
        # 解码器路径
        self.upconv4 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder4 = DoubleConv(256, 128)
        
        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder3 = DoubleConv(128, 64)
        
        self.upconv2 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.decoder2 = DoubleConv(128, 64)
        
        # 最终输出层
        self.outc = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 编码器
        x1 = self.inc(x)   # [B, 64, H, W]
        e1 = self.encoder1(x)   # [B, 64, H/2, W/2]
        e1 = self.se1(e1)
        e2 = self.encoder2(e1)  # [B, 64, H/2, W/2]
        e2 = self.se2(e2)
        e3 = self.encoder3(e2)  # [B, 128, H/4, W/4]
        e3 = self.se3(e3)
        e4 = self.encoder4(e3)  # [B, 256, H/8, W/8]
        e4 = self.se4(e4)
        
        # 解码器
        d4 = self.upconv4(e4)   # [B, 128, H/4, W/4]
        # 对齐尺寸
        if d4.size() != e3.size():
            d4 = F.interpolate(d4, size=e3.size()[2:], mode='bilinear', align_corners=False)
        d4 = torch.cat([d4, e3], dim=1)  # [B, 256, H/4, W/4]
        d4 = self.decoder4(d4)
        
        d3 = self.upconv3(d4)   # [B, 64, H/2, W/2]
        if d3.size() != e2.size():
            d3 = F.interpolate(d3, size=e2.size()[2:], mode='bilinear', align_corners=False)
        d3 = torch.cat([d3, e2], dim=1)  # [B, 128, H/2, W/2]
        d3 = self.decoder3(d3)
        
        d2 = self.upconv2(d3)   # [B, 64, H, W]
        if d2.size() != x1.size():
            d2 = F.interpolate(d2, size=x1.size()[2:], mode='bilinear', align_corners=False)
        d2 = torch.cat([d2, x1], dim=1)  # [B, 128, H, W]
        d2 = self.decoder2(d2)
        
        # 输出层
        out = self.outc(d2)
        return out

class SSIMLoss(nn.Module):
    """SSIM损失函数"""
    def __init__(self, window_size=11, size_average=True):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self._create_window(window_size)

    def _create_window(self, window_size):
        _1D_window = torch.Tensor([1.0 / window_size] * window_size).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(1, 1, window_size, window_size)
        return window

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        window = self.window.to(img1.device)
        window = window.expand(channel, 1, self.window_size, self.window_size)
        
        mu1 = F.conv2d(img1, window, padding=self.window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size//2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1*mu2

        sigma1_sq = F.conv2d(img1*img1, window, padding=self.window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding=self.window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding=self.window_size//2, groups=channel) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
        return 1 - ssim_map.mean()