# Version-2.0/Train.py
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from CoreModel import ResNetDeepfakeDetector

# ⚡ Added save_path argument to protect your ASVspoof weights
def train_resnet_model(train_loader, epochs=1, save_path="resnet_deepfake_detector_v2.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNetDeepfakeDetector().to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler('cuda')

    print(f"\n🚀 Launching Version 2.0 ResNet Engine Core on: {device}")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_steps = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch")
        
        for step, (waveforms, labels) in enumerate(progress_bar):
            waveforms = waveforms.to(device)
            labels = labels.to(device).unsqueeze(1)
            
            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast('cuda'):
                outputs = model(waveforms)
                loss = criterion(outputs, labels)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            total_steps += 1
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})
            
        avg_loss = total_loss / total_steps if total_steps > 0 else 0.0
        print(f"✨ Epoch {epoch+1:02d} Complete | Average Loss: {avg_loss:.4f}")
        
        # ⚡ Dynamically save the matrix
        torch.save(model.state_dict(), save_path)
        print(f"💾 Saved Version 2.0 Weights matrix to: '{save_path}'")
        torch.cuda.empty_cache()