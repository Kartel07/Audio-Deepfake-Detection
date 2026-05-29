from sklearn.metrics import roc_curve
import numpy as np
import torch
import pandas as pd
import os
from tqdm import tqdm

def evaluate_resnet_eer(model, test_loader, export_csv_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    all_scores, all_labels = [], []

    print("\n🚀 Commencing Version 2.0 ResNet Verification Scan...")
    with torch.no_grad():
        for waveforms, labels in tqdm(test_loader, desc="Evaluating Test Tracks"):
            waveforms = waveforms.to(device)
            with torch.amp.autocast('cuda'):
                logits = model(waveforms)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_scores.extend(probs.flatten())
            all_labels.extend(labels.numpy().flatten())

    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)

    fpr, tpr, thresholds = roc_curve(all_labels, all_scores, pos_label=1)
    idx = np.nanargmin(np.absolute((fpr - (1 - tpr))))
    eer, threshold = fpr[idx], thresholds[idx]
    
    print(f"EER: {eer*100:.2f}% | Threshold: {threshold:.4f}\n")
    
    if export_csv_path:
        os.makedirs(os.path.dirname(export_csv_path), exist_ok=True)
        pd.DataFrame({'True_Label': all_labels, 'Predicted_Score': all_scores}).to_csv(export_csv_path, index=False)
        
    return eer, threshold