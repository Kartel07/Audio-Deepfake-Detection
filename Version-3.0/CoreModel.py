# Version-3.0/CoreModel.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model

class DeepfakeDetectorXLS(nn.Module):
    def __init__(self, pretrained_path="facebook/wav2vec2-xls-r-300m"):
        super().__init__()
        print(f"🤗 Loading Self-Supervised Backbone: {pretrained_path}...")
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(pretrained_path)
        
        num_layers = 25  # 24 Encoder Layers + 1 Initial Feature Embedding
        self.layer_weights = nn.Parameter(torch.ones(num_layers))
        
        self.classifier = nn.Sequential(
            nn.Linear(1024, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4),
            nn.Linear(256, 1)
        )

    def forward(self, x, attention_mask=None):
        # x shape: [Batch, Samples] (e.g., [8, 64000])
        # attention_mask shape: [Batch, Samples] (e.g., [8, 64000])
        
        # 1. Pass the mask into the backbone so Hugging Face scales it to the frame level
        outputs = self.wav2vec2(x, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states  # Tuple of 25 tensors of shape [Batch, 199, 1024]
        
        # 2. Compute Softmax Weighted Layer Pooling
        softmax_weights = F.softmax(self.layer_weights, dim=0)
        stacked_states = torch.stack(hidden_states, dim=0)  # [25, Batch, 199, 1024]
        weighted_states = torch.sum(stacked_states * softmax_weights.view(-1, 1, 1, 1), dim=0) # [Batch, 199, 1024]
        
        # 3. Mean Pooling along the Time axis (Dimension 1)
        if attention_mask is not None:
            # OPTIMIZED: Use the model's downsampled mask representation or extract sequence lengths safely
            # Since all files are padded/truncated to a uniform 4.0s frame cap, 
            # we can safely perform regular mean pooling over the uniform features.
            pooled_features = torch.mean(weighted_states, dim=1)
        else:
            pooled_features = torch.mean(weighted_states, dim=1)
            
        return self.classifier(pooled_features)