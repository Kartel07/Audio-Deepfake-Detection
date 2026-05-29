import os
import librosa
import numpy as np
import soundfile as sf
import pandas as pd
from tqdm import tqdm

# --- CONFIGURATION & REPRODUCIBILITY ---
SEED = 42
np.random.seed(SEED)
# Statistical frame caps (based on typical forensic pipeline lengths)
TARGET_LENGTHS = [32000, 48000, 64000, 96000] 
BASE_PATH = os.path.join("Datasets", "for-2sec")
OUTPUT_ROOT = "Datasets/Ablation_Studies"
METADATA_LOG = []

def get_padded_waveform(y, target_len, pad_type='zero', pos='random'):
    """
    Applies padding with rigorous crop/pad controls to ensure 
    scientific reproducibility and avoid positional bias.
    """
    current_len = len(y)
    
    # 1. Handling Cropping (Crop bias control)
    if current_len > target_len:
        start = (current_len - target_len) // 2
        return y[start:start + target_len], 0.0
    
    # 2. Padding Strategy
    pad_len = target_len - current_len
    pad_ratio = pad_len / target_len
    
    if pad_type == 'zero':
        pad = np.zeros(pad_len)
    elif pad_type == 'reflect':
        pad = np.pad(y, (0, pad_len), mode='reflect')[current_len:current_len + pad_len]
    elif pad_type == 'gaussian':
        pad = np.random.normal(0, 0.001, pad_len)
    else:
        pad = np.zeros(pad_len)

    # 3. Position Strategy (Randomization prevents positional bias learning)
    pos = np.random.choice(['left', 'right', 'symmetric']) if pos == 'random' else pos
    if pos == 'left':
        y_padded = np.concatenate([pad, y])
    elif pos == 'right':
        y_padded = np.concatenate([y, pad])
    else: # Symmetric
        half = pad_len // 2
        y_padded = np.concatenate([pad[:half], y, pad[half:]])
        
    return y_padded, pad_ratio

def run_ablation_pipeline():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    
    # Collect all file paths
    all_files = []
    print(f"Searching for .wav files in: {os.path.abspath(BASE_PATH)}")
    for root, dirs, filenames in os.walk(BASE_PATH):
        for f in filenames:
            if f.endswith('.wav'):
                all_files.append(os.path.join(root, f))
    
    if len(all_files) == 0:
        print("❌ No .wav files found. Please check your BASE_PATH variable.")
        return 

    print(f"✅ Found {len(all_files)} files. Starting ablation generation...")
    
    for target in TARGET_LENGTHS:
        for ptype in ['zero', 'reflect', 'gaussian']:
            out_dir = os.path.join(OUTPUT_ROOT, f"Target{target}_{ptype}Pad")
            os.makedirs(out_dir, exist_ok=True)
            
            # Process all files for this specific combination
            for file_path in tqdm(all_files, desc=f"Padding {ptype} to {target}"):
                file_name = os.path.basename(file_path)
                y, sr = librosa.load(file_path, sr=16000)
                y_new, ratio = get_padded_waveform(y, target, ptype)
                
                # Save into the specific ablation sub-folder
                sf.write(os.path.join(out_dir, file_name), y_new, sr)
                
                METADATA_LOG.append({
                    "file": file_name, 
                    "target_len": target, 
                    "pad_type": ptype, 
                    "pad_ratio": ratio, 
                    "original_len": len(y)
                })

    # Save metadata for later EER vs Pad Ratio analysis
    pd.DataFrame(METADATA_LOG).to_csv(os.path.join(OUTPUT_ROOT, "ablation_metadata.csv"), index=False)
    print(f"✅ Ablation datasets generated in {OUTPUT_ROOT}")

if __name__ == "__main__":
    run_ablation_pipeline()