# singletest.py
import librosa
import numpy as np
from CoreModel import DeepfakeDetectorDSP

def test_single_file(audio_path, model_path="best_deepfake_detector.pt"):
    """
    Runs an interactive acoustic diagnostic scan on an individual audio track.
    """
    model = DeepfakeDetectorDSP()
    model.load_model(model_path)
    
    try:
        audio, _ = librosa.load(audio_path, sr=16000)
        target_len = 64000
        if len(audio) > target_len: 
            audio = audio[:target_len]
        else: 
            audio = np.pad(audio, (0, target_len - len(audio)), 'constant')
        
        feats = model.extract_acoustic_features(audio)
        prob = model.classifier.predict_proba(feats.reshape(1, -1))[0][1]
        
        # Translate the probability profile into a standard logit value
        prob_clipped = np.clip(prob, 1e-7, 1.0 - 1e-7)
        logit_val = np.log(prob_clipped / (1.0 - prob_clipped))
        
        # Direct mathematical verification mapping: Logit >= 0 is a deepfake clone
        result = "FAKE/SPOOF" if logit_val >= 0.0 else "REAL/BONAFIDE"
        confidence = prob * 100 if result == "FAKE/SPOOF" else (1.0 - prob) * 100
        
        print(f"\n--- Forensic Analysis Output ---")
        print(f"Target Track Location: {audio_path}")
        print(f"System Verdict: {result}")
        print(f"Confidence Level Match: {confidence:.2f}%")
        print(f"Raw Score Logit: {logit_val:.4f}")
        print(f"Spectral Kurtosis Value: {feats[-2]:.4f}")
    except Exception as e:
        print(f"Error executing file analysis: {e}")

if __name__ == "__main__":
    target_sample = r"C:\Users\karti\Downloads\real.wav"
    test_single_file(target_sample)