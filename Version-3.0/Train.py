# Version-3.0/Train.py
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import queue
import threading
from CoreModel import DeepfakeDetectorXLS

def _prefetch_worker(loader, data_queue, device):
    """Continually streams data tensors from CPU workers directly into VRAM asynchronously."""
    for audio, masks, labels in loader:
        # Send to GPU with non_blocking=True to prevent CPU thread stalls
        audio = audio.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).unsqueeze(1)
        data_queue.put((audio, masks, labels))
    data_queue.put(None) # Sentinel to close stream cleanly

# ⚡ Added save_path argument
def train_transformer_model(train_loader, epochs=1, save_path="transformer_deepfake_detector_v3.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepfakeDetectorXLS().to(device)
    
    # --- HARDENING STEP: LAYER FREEZING ---
    print("🔒 Freezing early layers to preserve general speech representations...")
    if hasattr(model.wav2vec2, 'feature_extractor'):
        for param in model.wav2vec2.feature_extractor.parameters():
            param.requires_grad = False
            
    if hasattr(model.wav2vec2, 'encoder') and hasattr(model.wav2vec2.encoder, 'layers'):
        for i in range(12): # Freeze first half of the 24 Transformer layers
            for param in model.wav2vec2.encoder.layers[i].parameters():
                param.requires_grad = False

    active_params = [p for p in model.parameters() if p.requires_grad]
    
    optimizer = optim.AdamW(active_params, lr=1e-5, weight_decay=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler('cuda')
    
    accumulation_steps = 4

    print(f"\n🚀 Launching Asynchronous Prefetched Fine-Tuning on: {device}")
    for epoch in range(epochs):
        model.train()
        total_loss, valid_batches = 0, 0
        optimizer.zero_grad(set_to_none=True)
        
        # Initialize background RAM prefetcher queue
        data_queue = queue.Queue(maxsize=16)
        prefetch_thread = threading.Thread(target=_prefetch_worker, args=(train_loader, data_queue, device))
        prefetch_thread.daemon = True
        prefetch_thread.start()
        
        progress_bar = tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}", unit="batch")
        step = 0
        
        while True:
            batch_data = data_queue.get()
            if batch_data is None:
                break
                
            audio_clean, masks_clean, labels_clean = batch_data
            
            with torch.amp.autocast('cuda'):
                outputs = model(audio_clean, attention_mask=masks_clean)
                loss = criterion(outputs, labels_clean) / accumulation_steps
                
                # Keep backward graph pass inside fast FP16 execution routing context
                scaler.scale(loss).backward()
            
            if (step + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                
            total_loss += loss.item() * accumulation_steps
            valid_batches += 1
            
            if step % 4 == 0:
                progress_bar.set_postfix({"Loss": f"{(loss.item() * accumulation_steps):.4f}"})
            progress_bar.update(1)
            step += 1
            
        progress_bar.close()
        prefetch_thread.join()
        
        avg_loss = total_loss / valid_batches if valid_batches > 0 else 0.0
        print(f"✨ Epoch {epoch+1:02d} Complete | Average Loss: {avg_loss:.4f}")
        
        # ⚡ Dynamically save the weights
        torch.save(model.state_dict(), save_path)
        print(f"💾 Saved Flagship Version 3.0 Weights to: '{save_path}'")
        torch.cuda.empty_cache()