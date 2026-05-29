import torch
import librosa
import numpy as np
from CoreModel import DeepfakeDetectorXLS

def test_single_file(audio_path, model_path="best_deepfake_detector.pt"):
    device = torch.device("cpu") # Force CPU for stability
    
    # 1. Load the model
    model = DeepfakeDetectorXLS().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 2. Preprocess (16kHz, 2-second window to match your 32k training)
    try:
        audio, _ = librosa.load(audio_path, sr=16000)
        target_length = 32000 # 2 seconds
        
        if len(audio) > target_length:
            audio = audio[:target_length]
        else:
            audio = np.pad(audio, (0, target_length - len(audio)), 'constant')
        
        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).to(device)
        
       # 3. Inference
        with torch.no_grad():
            logits = model(audio_tensor)
            # LOGITS tell us the raw "opinion" before squashing to 0-1
            raw_val = logits.item()
            probability = torch.sigmoid(logits).item() 
            
        # 4. Corrected Logic based on Logit -11.38
        # Logic: Negative Logit (< 0.5 Prob) = FAKE
        #        Positive Logit (> 0.5 Prob) = REAL
        
        if raw_val < -11.3845:
            result = "FAKE/SPOOF"
            # Confidence is high when probability is close to 0
            confidence = (1 - probability) 
        else:
            result = "REAL/BONAFIDE"
            # Confidence is high when probability is close to 1
            confidence = probability

        print(f"\n--- Detection Result ---")
        print(f"File: {audio_path}")
        print(f"Prediction: {result}")
        if(result == "FAKE/SPOOF"):
            print(f"Confidence: {(confidence):.2f}%\n Logit Score: {raw_val:.4f}")
        else:
            print(f"Confidence: {(1-confidence):.2f}%, Logit Score: {raw_val:.4f}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # file that ACTUALLY exists on desktop
    target = r"C:\Users\karti\Downloads\real.wav" 
    test_single_file(target)