import warnings
from sklearn.metrics import roc_curve
import numpy as np
import torch
import pandas as pd
import os
from tqdm import tqdm
import queue
import threading

warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

def _prefetch_worker(loader, data_queue):
    """Background worker thread optimized for streaming raw 1D waves cleanly into memory."""
    for batch in loader: 
        data_queue.put(batch)
    data_queue.put(None)

def evaluate_rawnet_eer(model, test_loader, export_csv_path=None):
    """
    Computes rigorous evaluation metrics for the RawNet2 network
    using asynchronous prefetching to maximize local desktop I/O throughput.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    all_scores, all_labels = [], []

    print(f"\n[1/2] Spinning up Asynchronous Audio Queue Prefetcher...")
    data_queue = queue.Queue(maxsize=32)
    prefetch_thread = threading.Thread(target=_prefetch_worker, args=(test_loader, data_queue))
    prefetch_thread.daemon = True
    prefetch_thread.start()

    test_progress = tqdm(total=len(test_loader), desc="Evaluating RawNet2 Audio Tracks", unit="batch")

    with torch.no_grad():
        while True:
            batch_data = data_queue.get()
            if batch_data is None:
                break
                
            # Safely unpack
            if len(batch_data) >= 2:
                audio = batch_data[0]
                labels = batch_data[1]
            else:
                raise ValueError("Unexpected batch format.")

            audio = audio.to(device, non_blocking=True)
            
            with torch.amp.autocast('cuda'):
                logits = model(audio)
                
            probs = torch.sigmoid(logits).cpu().numpy()
            
            all_scores.extend(probs.flatten())
            all_labels.extend(labels.numpy().flatten())
            test_progress.update(1)

    test_progress.close()
    prefetch_thread.join()

    print("\n[2/2] Running ROC Curve Metrics Calculation Ledger...")
    all_labels, all_scores = np.array(all_labels), np.array(all_scores)
    fpr, tpr, thresholds = roc_curve(all_labels, all_scores, pos_label=1)
    fnr = 1 - tpr
    idx = np.nanargmin(np.absolute((fpr - fnr)))
    eer, threshold = fpr[idx], thresholds[idx]
    
    print(f"\n📈 --- Version 4.0 RawNet2 SincNet EER Summary ---")
    print(f"Equal Error Rate (EER): {eer*100:.2f}%")
    print(f"Optimal Probability Threshold Target: {threshold:.4f}\n")
    
    # --- ADDED: EXPORT LOGIC ---
    if export_csv_path:
        os.makedirs(os.path.dirname(export_csv_path), exist_ok=True)
        df = pd.DataFrame({'True_Label': all_labels, 'Predicted_Score': all_scores})
        df.to_csv(export_csv_path, index=False)
        print(f"✅ Raw prediction scores exported to: {export_csv_path}")

    return eer, threshold