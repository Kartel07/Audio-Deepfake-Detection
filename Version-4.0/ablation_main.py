import os
import sys
import torch
import pandas as pd
from torch.utils.data import DataLoader

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
if root_dir not in sys.path:
    sys.path.append(root_dir)

from dataPipeline_FoR import FakeOrReal2SecDataset
from CoreModel import RawNet2Detector
from Test import evaluate_rawnet_eer

ABLATION_ROOT = os.path.join(root_dir, "Datasets", "Ablation_Studies")
WEIGHTS_FILE = os.path.join(current_dir, "rawnet2_deepfake_detector_v4_FoR.pt")
BATCH_SIZE = 32

def run_ablation_matrix():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RawNet2Detector().to(device)
    
    if os.path.exists(WEIGHTS_FILE):
        model.load_state_dict(torch.load(WEIGHTS_FILE, map_location=device, weights_only=False))
    else:
        print("❌ Baseline weights not found.")
        return

    results = []
    folders = [f for f in os.listdir(ABLATION_ROOT) if os.path.isdir(os.path.join(ABLATION_ROOT, f))]
    
    for folder in folders:
        folder_path = os.path.join(ABLATION_ROOT, folder)
        # Use split=None to force flat scan
        test_dataset = FakeOrReal2SecDataset(folder_path, split=None)
        
        if len(test_dataset) == 0:
            print(f"⚠️ Skipping {folder}: No valid tracks found.")
            continue
            
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        
        try:
            eer = evaluate_rawnet_eer(model, test_loader)
            results.append({"folder": folder, "eer": eer})
        except Exception as e:
            print(f"❌ Error evaluating {folder}: {e}")

    pd.DataFrame(results).to_csv("ablation_results_summary.csv", index=False)
    print("\n✅ Ablation Study Complete.")

if __name__ == "__main__":
    run_ablation_matrix()