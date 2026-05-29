# Version-4.0/Train.py
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from CoreModel import RawNet2Detector

# ⚡ Added save_path and removed the 8.7 pos_weight since FoR is 1:1 balanced
def train_rawnet_model(train_loader, epochs=5, save_path="rawnet2_deepfake_detector_v4_balanced.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RawNet2Detector().to(device)
    
    # ⚡ Pure, unweighted BCE loss for a perfectly balanced dataset
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    scaler = torch.amp.GradScaler('cuda')

    print(f"\n🚀 Launching Version 4.0 RawNet2 Engine Core on: {device}")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_steps = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch")
        
        for step, (waveforms, labels) in enumerate(progress_bar):
            # ⚡ Stream directly to GPU asynchronously
            waveforms = waveforms.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).unsqueeze(1)
            
            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast('cuda'):
                outputs = model(waveforms)
                loss = criterion(outputs, labels)
                
            scaler.scale(loss).backward()
            
            # Gradient clipping to protect SincNet filters
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            total_steps += 1
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})
            
        avg_loss = total_loss / total_steps if total_steps > 0 else 0.0
        print(f"✨ Epoch {epoch+1:02d} Complete | Average Loss: {avg_loss:.4f}")
        
        # ⚡ Dynamically save the weights to the target requested by main.py
        torch.save(model.state_dict(), save_path)
        print(f"💾 Saved Version 4.0 Weights to: '{save_path}'")
        torch.cuda.empty_cache()