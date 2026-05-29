# Version-1.0/Test.py
from sklearn.metrics import roc_curve
import numpy as np
import torch
import pandas as pd
import os
from tqdm import tqdm

def evaluate_baseline_eer(model, test_loader, export_csv_path=None):
    """
    Computes rigorous system verification profiles and optionally exports raw
    prediction scores for academic plotting (DET curves, Histograms).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    all_scores, all_labels = [], []

    print("\n🚀 Commencing Version 1.0 Verification Scan...")
    
    with torch.no_grad():
        for waveforms, labels in tqdm(test_loader, desc="Evaluating Test Tracks"):
            waveforms = waveforms.to(device)
            
            # Fast FP16 mixed-precision inference
            with torch.amp.autocast('cuda'):
                logits = model(waveforms)
                
            # Map logits to probabilities
            probs = torch.sigmoid(logits).cpu().numpy()
            
            all_scores.extend(probs.flatten())
            all_labels.extend(labels.numpy().flatten())

    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)

    # Locate Equal Error Rate
    fpr, tpr, thresholds = roc_curve(all_labels, all_scores, pos_label=1)
    fnr = 1 - tpr
    
    idx = np.nanargmin(np.absolute((fpr - fnr)))
    eer, threshold = fpr[idx], thresholds[idx]
    
    print(f"\n📈 --- Version 1.0 Validation Profile Summary ---")
    print(f"Evaluation EER: {eer*100:.2f}% | Optimal Decision Cutoff: {threshold:.4f}\n")
    
    # --- EXPORT RAW SCORES FOR PAPER 1 ---
    if export_csv_path:
        os.makedirs(os.path.dirname(export_csv_path), exist_ok=True)
        df = pd.DataFrame({'True_Label': all_labels, 'Predicted_Score': all_scores})
        df.to_csv(export_csv_path, index=False)
        print(f"✅ Raw prediction scores exported to: {export_csv_path}")
        
    return eer, threshold