# Version-1.0/Train.py
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from CoreModel import BaselineDeepfakeDetector

# ⚡ Added save_path argument with a default fallback to preserve backward compatibility
def train_baseline_model(train_loader, epochs=3, save_path="baseline_deepfake_detector_v1_balanced.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BaselineDeepfakeDetector().to(device)
    
    # Standard baseline tuning setup
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler('cuda')

    print(f"\n🚀 Launching Version 1.0 Baseline Training Core on: {device}")
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
        
        # ⚡ Now dynamically saves to whichever dataset profile main.py requested!
        torch.save(model.state_dict(), save_path)
        print(f"💾 Saved Version 1.0 Weights to: '{save_path}'")
        torch.cuda.empty_cache()