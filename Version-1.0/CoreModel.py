# Version-1.0/CoreModel.py
import torch
import torch.nn as nn
import torchaudio.transforms as T

class BaselineDeepfakeDetector(nn.Module):
    """
    Version 1.0 Baseline: Converts raw audio to Log-Mel Spectrograms 
    and classifies them using an optimized 2D Convolutional Network.
    """
    def __init__(self, sample_rate=16000, n_mels=80):
        super().__init__()
        
        # 1. On-the-fly Mel-Spectrogram Extraction Layer
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            win_length=400,
            hop_length=160,
            n_mels=n_mels
        )
        
        # 2. Convolutional Feature Extraction Network
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Downsample timeline/frequencies
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)) # Flattens temporal variations cleanly
        )
        
        # 3. Dense Linear Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.5), # Regularization to combat domain overfitting
            nn.Linear(128, 1) # Output Logit: >0 = Spoof, <=0 = Bonafide
        )

    def forward(self, x):
        # x shape: [Batch, Samples]
        with torch.no_grad():
            # Convert to Mel-Spectrogram and apply Log scale for acoustic stability
            mel_spec = self.mel_transform(x)
            log_mel = torch.log(mel_spec + 1e-6)
            
        # Add explicit Channel dimension [Batch, 1, Mels, Time]
        log_mel = log_mel.unsqueeze(1)
        
        features = self.feature_extractor(log_mel)
        features = features.view(features.size(0), -1) # Flatten
        
        return self.classifier(features)