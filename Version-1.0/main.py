import os
import sys
import torch
import argparse
from torch.utils.data import DataLoader

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
if root_dir not in sys.path:
    sys.path.append(root_dir)

from dataPipeline import ASVspoof5BaselineDataset
from dataPipeline_FoR import FakeOrReal2SecDataset
from CoreModel import BaselineDeepfakeDetector
from Train import train_baseline_model
from Test import evaluate_baseline_eer

def parse_args():
    parser = argparse.ArgumentParser(description="Version 1.0 CNN Baseline Execution Matrix")
    parser.add_name = "V1_CNN"
    parser.add_argument("--mode", type=str, choices=["train", "test"], required=True, help="Execution mode")
    parser.add_argument("--train_on", type=str, choices=["FoR", "ASVspoof5"], required=True, help="Training domain")
    parser.add_argument("--test_on", type=str, choices=["FoR", "ASVspoof5"], default="ASVspoof5", help="Testing domain")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    return parser.parse_args()

def main():
    args = parse_args()
    print(f"\n--- Starting Version 1.0 Baseline CNN System Matrix ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Dynamic weight matrix paths
    weight_filename = f"baseline_deepfake_detector_v1_{args.train_on}.pt"
    weights_path = os.path.join(current_dir, weight_filename)

    if args.mode == "train":
        print(f"🎯 Action: TRAINING on {args.train_on}")
        if args.train_on == "ASVspoof5":
            train_dataset = ASVspoof5BaselineDataset(os.path.join(current_dir, "ASVspoof5.train.tsv"), os.path.join(root_dir, "Datasets", "flac_T"))
        else:
            train_dataset = FakeOrReal2SecDataset(os.path.join(root_dir, "Datasets", "for-2sec"), split="training")
            
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
        train_baseline_model(train_loader, epochs=args.epochs, save_path=weights_path)

    elif args.mode == "test":
        print(f"🎯 Action: TESTING | Trained on: {args.train_on} -> Evaluating on: {args.test_on}")
        if args.test_on == "ASVspoof5":
            test_dataset = ASVspoof5BaselineDataset(os.path.join(current_dir, "ASVspoof5.dev.track_1.tsv"), os.path.join(root_dir, "Datasets", "flac_D"))
        else:
            test_dataset = FakeOrReal2SecDataset(os.path.join(root_dir, "Datasets", "for-2sec"), split="validation")
            
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
        model = BaselineDeepfakeDetector().to(device)
        
        if os.path.exists(weights_path):
            model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=False))
            print("✅ Weight matrix loaded successfully.")
        else:
            print(f"❌ Error: Weights path '{weights_path}' missing. Run training first.")
            return

        export_path = os.path.join(root_dir, "Paper1_Data_Exports", f"raw_scores_V1_Train{args.train_on}_Test{args.test_on}.csv")
        os.makedirs(os.path.dirname(export_path), exist_ok=True)
        evaluate_baseline_eer(model, test_loader, export_csv_path=export_path)

if __name__ == "__main__":
    main()