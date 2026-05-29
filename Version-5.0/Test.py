# Version-5.0/Test.py
from sklearn.metrics import roc_curve
import numpy as np
import torch
import pandas as pd
import os
from tqdm import tqdm

def evaluate_orthogonal_fusion_eer(model_v2, model_v3, test_loader, export_csv_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_v2.eval()
    model_v3.eval()
    
    all_scores, all_labels = [], []

    print("\n🚀 Commencing Version 5.0 Orthogonal Fusion Verification Scan...")
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating Dual-Path Tracks"):
            # Unpack safely
            if len(batch) == 3:
                waveforms, masks, labels = batch
            elif len(batch) == 2:
                waveforms, labels = batch
                masks = torch.ones_like(waveforms)
            else:
                raise ValueError("Unexpected batch format.")

            waveforms = waveforms.to(device)
            masks = masks.to(device)
            
            with torch.amp.autocast('cuda'):
                # ResNet Inference
                logits_v2 = model_v2(waveforms)
                probs_v2 = torch.sigmoid(logits_v2)
                
                # Transformer Inference
                logits_v3 = model_v3(waveforms, attention_mask=masks)
                probs_v3 = torch.sigmoid(logits_v3)
                
                # Late Probability Fusion (Mean Averaging)
                fused_probs = (probs_v2 + probs_v3) / 2.0
                
            all_scores.extend(fused_probs.cpu().numpy().flatten())
            all_labels.extend(labels.numpy().flatten())

    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)
    
    fpr, tpr, thresholds = roc_curve(all_labels, all_scores, pos_label=1)
    fnr = 1 - tpr
    idx = np.nanargmin(np.absolute((fpr - fnr)))
    eer, threshold = fpr[idx], thresholds[idx]
    
    print(f"\n📈 --- Version 5.0 Fusion Validation Profile Summary ---")
    print(f"Evaluation EER: {eer*100:.2f}% | Optimal Decision Cutoff: {threshold:.4f}\n")
    
    if export_csv_path:
        os.makedirs(os.path.dirname(export_csv_path), exist_ok=True)
        pd.DataFrame({'True_Label': all_labels, 'Predicted_Score': all_scores}).to_csv(export_csv_path, index=False)
        print(f"✅ Raw prediction scores exported to: {export_csv_path}")
        
    return eer, threshold