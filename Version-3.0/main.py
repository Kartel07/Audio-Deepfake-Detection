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
from CoreModel import ResNetDeepfakeDetector, XLSRTransformerDetector

# Ensure this matches the exact name of your fusion evaluation script
from Test import evaluate_orthogonal_fusion_eer

def parse_args():
    parser = argparse.ArgumentParser(description="Version 5.0 Orthogonal Fusion Engine Matrix")
    
    # Notice there is no --mode flag here, because V5 is strictly an evaluation ensemble.
    parser.add_argument("--train_on", type=str, choices=["FoR", "ASVspoof5"], required=True, 
                        help="Which foundational domain weights to load for V2 and V3.")
    parser.add_argument("--test_on", type=str, choices=["FoR", "ASVspoof5"], default="ASVspoof5", 
                        help="The target dataset to evaluate the fusion on.")
    parser.add_argument("--batch_size", type=int, default=16, 
                        help="Batch size (kept small to fit both V2 and V3 in 6GB VRAM)")
    
    return parser.parse_args()

def main():
    args = parse_args()
    print(f"\n--- Starting Version 5.0 Orthogonal Fusion Verification Pipeline ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Dynamically resolve component weights paths based on the chosen training configuration
    v2_weights = os.path.join(root_dir, "Version-2.0", f"resnet_deepfake_detector_v2_{args.train_on}.pt")
    v3_weights = os.path.join(root_dir, "Version-3.0", f"xlsr_transformer_detector_v3_{args.train_on}.pt")

    print(f"🎯 Action: EVALUATING FUSION | Assembled From: {args.train_on} -> Evaluating on: {args.test_on}")
    
    # 1. Load the Target Dataset
    if args.test_on == "ASVspoof5":
        test_dataset = ASVspoof5BaselineDataset(os.path.join(root_dir, "Version-1.0", "ASVspoof5.dev.track_1.tsv"), 
                                                os.path.join(root_dir, "Datasets", "flac_D"))
    else:
        test_dataset = FakeOrReal2SecDataset(os.path.join(root_dir, "Datasets", "for-2sec"), split="validation")
        
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # 2. Initialize the Sub-Networks
    model_v2 = ResNetDeepfakeDetector().to(device)
    model_v3 = XLSRTransformerDetector().to(device)
    
    # 3. Load Foundational Weights Securely
    if os.path.exists(v2_weights) and os.path.exists(v3_weights):
        model_v2.load_state_dict(torch.load(v2_weights, map_location=device, weights_only=False))
        model_v3.load_state_dict(torch.load(v3_weights, map_location=device, weights_only=False))
        print("✅ Foundational V2 (ResNet) and V3 (Transformer) models successfully assembled in VRAM.")
    else:
        print(f"❌ Error: Missing component weights.")
        if not os.path.exists(v2_weights):
            print(f"   -> Missing: {v2_weights}")
        if not os.path.exists(v3_weights):
            print(f"   -> Missing: {v3_weights}")
        print(f"Please ensure both V2 and V3 are trained on '{args.train_on}' before running fusion.")
        return

    # 4. Prepare the Export Path for the Raw Scores CSV
    export_path = os.path.join(root_dir, "Paper1_Data_Exports", f"raw_scores_V5_Train{args.train_on}_Test{args.test_on}.csv")
    os.makedirs(os.path.dirname(export_path), exist_ok=True)
    
    # 5. Execute Fusion Inference
    evaluate_orthogonal_fusion_eer(model_v2, model_v3, test_loader, export_csv_path=export_path)

if __name__ == "__main__":
    main()