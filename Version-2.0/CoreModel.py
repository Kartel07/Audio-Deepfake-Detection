# Version-2.0/CoreModel.py
import torch
import torch.nn as nn

class ResidualBlock2D(nn.Module):
    """
    Standard ResNet Residual Block with skip connections to prevent 
    gradient degradation across deeper networks.
    """
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
            
        out += residual
        out = self.relu(out)
        return out

class ResNetDeepfakeDetector(nn.Module):
    """
    Version 2.0 Advanced Baseline: Deep Residual Network (ResNet-34 style) 
    designed to extract deep spatial acoustic signatures.
    """
    def __init__(self):
        super().__init__()
        # Initial spectro-temporal feature extraction layer
        self.in_channels = 32
        self.conv1 = nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet Layer Blocks
        self.layer1 = self._make_layer(32, 2)
        self.layer2 = self._make_layer(64, 2, stride=2)
        self.layer3 = self._make_layer(128, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(128, 1)

    def _make_layer(self, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        layers = []
        layers.append(ResidualBlock2D(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(ResidualBlock2D(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        # On-the-fly Spectrogram conversion mapping layer representation
        # x input shape: [Batch, Samples] -> converted internally to a 2D feature patch
        with torch.no_grad():
            # Linear spectrogram conversion acting as placeholder for high-res geometric features
            window = torch.hann_window(400, device=x.device)
            stft = torch.stft(x, n_fft=512, hop_length=160, win_length=400, window=window, return_complex=True)
            mag = torch.abs(stft)
            log_spec = torch.log(mag + 1e-6)
            
        # Add channel dimension: [Batch, 1, Freq, Time]
        x = log_spec.unsqueeze(1)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)