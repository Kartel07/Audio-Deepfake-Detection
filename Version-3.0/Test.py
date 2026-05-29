from sklearn.metrics import roc_curve
import numpy as np
import torch
import pandas as pd
import os
from tqdm import tqdm

def evaluate_transformer_eer(model, test_loader, export_csv_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    all_scores, all_labels = [], []

    print("\n🚀 Commencing Version 3.0 XLS-R Verification Scan...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating Test Tracks"):
            # Dynamically unpack based on dataset format
            if len(batch) == 3:
                waveforms, masks, labels = batch
            elif len(batch) == 2:
                waveforms, labels = batch
                # Create a default mask of all ones for datasets that don't provide one
                masks = torch.ones_like(waveforms)
            else:
                raise ValueError(f"Unexpected batch format with {len(batch)} elements.")

            waveforms = waveforms.to(device)
            masks = masks.to(device)
            
            with torch.amp.autocast('cuda'):
                # Pass the mask to the model
                logits = model(waveforms, attention_mask=masks)
                
            probs = torch.sigmoid(logits).cpu().numpy()
            all_scores.extend(probs.flatten())
            all_labels.extend(labels.numpy().flatten())

    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)
    fpr, tpr, thresholds = roc_curve(all_labels, all_scores, pos_label=1)
    idx = np.nanargmin(np.absolute((fpr - (1 - tpr))))
    eer, threshold = fpr[idx], thresholds[idx]
    
    print(f"\n📈 --- Version 3.0 Validation Profile Summary ---")
    print(f"Evaluation EER: {eer*100:.2f}% | Optimal Decision Cutoff: {threshold:.4f}\n")
    
    if export_csv_path:
        os.makedirs(os.path.dirname(export_csv_path), exist_ok=True)
        pd.DataFrame({'True_Label': all_labels, 'Predicted_Score': all_scores}).to_csv(export_csv_path, index=False)
        print(f"✅ Raw prediction scores exported to: {export_csv_path}")
        
    return eer, threshold