# Version-4.0/CoreModel.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SincConv1D(nn.Module):
    """
    SincNet learnable 1D frontend band-pass filter layer.
    Learns low and high cutoff frequencies directly via backpropagation.
    """
    def __init__(self, out_channels, kernel_size, sample_rate=16000):
        super().__init__()
        if kernel_size % 2 == 0:
            kernel_size += 1  # Enforce odd kernel length for absolute symmetry
            
        self.sample_rate = sample_rate
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        
        # Initialize learnable frequency bands distributed across mel-scale space
        min_freq = 50
        max_freq = sample_rate / 2
        
        b1 = np.linspace(min_freq, max_freq - min_freq, out_channels)
        b2 = b1 + min_freq
        
        self.filt_b1 = nn.Parameter(torch.from_numpy(b1 / sample_rate).float().unsqueeze(1))
        self.filt_band = nn.Parameter(torch.from_numpy((b2 - b1) / sample_rate).float().unsqueeze(1))
        
        # Hamming window initialization vectors
        n_lin = torch.linspace(-((kernel_size - 1) / 2), ((kernel_size - 1) / 2), steps=kernel_size)
        self.register_buffer('window', 0.54 - 0.46 * torch.cos(2 * np.pi * (n_lin + (kernel_size - 1) / 2) / kernel_size))
        self.register_buffer('n', 2 * np.pi * n_lin)

    def forward(self, x):
        # x shape: [Batch, 1, Samples]
        device = x.device
        
        # Mirror frequencies to compute absolute high/low bands cleanly
        low = torch.abs(self.filt_b1)
        high = torch.clamp(low + torch.abs(self.filt_band), min=0, max=0.5)
        
        filters = []
        for i in range(self.out_channels):
            f_low = low[i] * self.sample_rate
            f_high = high[i] * self.sample_rate
            
            # Mathematical Sinc interpolation formula mapping
            sinc_high = torch.sinc(self.n * f_high / self.sample_rate)
            sinc_low = torch.sinc(self.n * f_low / self.sample_rate)
            
            # Compute final bandpass window filter configuration
            filt = (sinc_high - sinc_low) * self.window
            filters.append(filt / torch.norm(filt, p=2))
            
        filters = torch.stack(filters).unsqueeze(1).to(device) # [OutChannels, 1, KernelSize]
        return F.conv1d(x, filters, stride=3, padding=self.kernel_size // 2)

class ResidualBlock1D(nn.Module):
    """
    Upgraded 1D Residual processing block with fully activated non-linear streams.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        
        # Stream Layer 1
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.leaky_relu(out, 0.2)
        
        # Stream Layer 2
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.leaky_relu(out, 0.2) # ⚡ FIX: Activated non-linear mapping state before residual summation
        
        out += residual
        return out

class RawNet2Detector(nn.Module):
    """
    Version 4.0 Core Architecture: Direct Raw 1D Audio Waveform 
    Processing via Learnable SincNet Frontend + ResNet Backend.
    """
    def __init__(self):
        super().__init__()
        # SincNet learnable spectral analyzer frontend
        self.sinc = SincConv1D(out_channels=20, kernel_size=251)
        self.bn_init = nn.BatchNorm1d(20)
        
        # Feature processing feature extractor path
        self.layer1 = ResidualBlock1D(20, 32)
        self.layer2 = ResidualBlock1D(32, 64)
        self.layer3 = ResidualBlock1D(64, 128)
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # Dense classification framework head
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4),  # Strong anti-overfitting regularizer
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # Input x shape context expected: [Batch, Samples] (e.g., [Batch, 64000])
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add channel dimension: [Batch, 1, Samples]
            
        out = F.leaky_relu(self.bn_init(self.sinc(x)), 0.2)
        out = F.max_pool1d(out, kernel_size=3, stride=3)
        
        out = self.layer1(out)
        out = F.max_pool1d(out, kernel_size=3, stride=3)
        out = self.layer2(out)
        out = F.max_pool1d(out, kernel_size=3, stride=3)
        out = self.layer3(out)
        
        out = self.pool(out).flatten(1)
        return self.fc(out)