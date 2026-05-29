# Version-3.0/save_and_test.py
import os
import sys
import torch
from torch.utils.data import DataLoader

# Setup project root context path routing
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
if root_dir not in sys.path:
    sys.path.append(root_dir)

from dataPipeline import ASVspoof5TransformerDataset
from CoreModel import DeepfakeDetectorXLS
from Test import evaluate_transformer_eer

def main():
    print("--- Emergency Intercept: Evaluating 50% Trained Transformer ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    TEST_PROTOCOL = os.path.join(current_dir, "ASVspoof5.dev.track_1.tsv")
    TEST_AUDIO_DIR = os.path.join(root_dir, "Datasets", "flac_D")
    
    # Instantiate the dataset using the 2.0-second optimization window
    test_dataset = ASVspoof5TransformerDataset(TEST_PROTOCOL, TEST_AUDIO_DIR, max_duration=2.0)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=2, pin_memory=True)
    
    model = DeepfakeDetectorXLS().to(device)
    
    # Target the saved weight matrix file on disk
    weights_path = os.path.join(current_dir, "transformer_deepfake_detector_v3.pt")
    if os.path.exists(weights_path):
        # Using weights_only=False since we are loading standard state dict tensors
        model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=False))
        print("✅ Successfully loaded partially trained 50% weight matrices.")
    else:
        print("⚠️ No checkpoint found at 'transformer_deepfake_detector_v3.pt'. Check your directory path!")
        return

    # Run the evaluation loop from Test.py
    evaluate_transformer_eer(model, test_loader)

if __name__ == "__main__":
    main()