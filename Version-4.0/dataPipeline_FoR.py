# dataPipeline_FoR.py
import os
import json
import torch
import torchaudio
from torch.utils.data import Dataset

class FakeOrReal2SecDataset(Dataset):
    def __init__(self, root_dir, split=None):
        self.max_samples = 32000
        self.file_list = []
        
        # 1. Load the label map
        map_path = "label_map.json"
        if not os.path.exists(map_path):
             print(f"❌ Error: {map_path} not found. Please run create_label_map.py first.")
             return
             
        with open(map_path, "r") as f:
            self.label_map = json.load(f)
            
        # Path logic: handles nested splits or flat ablation folders
        target_path = os.path.join(root_dir, split) if split and os.path.exists(os.path.join(root_dir, split)) else root_dir
        
        # Crawl for audio files
        for root, _, files in os.walk(target_path):
            for f in files:
                if f.endswith(('.wav', '.flac')):
                    # 2. Lookup the label from the map using the filename
                    # If it's not found in the map, we skip it
                    if f in self.label_map:
                        label = self.label_map[f]
                        self.file_list.append((os.path.join(root, f), label))
                    else:
                        continue 
                        
        print(f"✅ Loaded {len(self.file_list)} valid tracks using label map.")

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        path, label = self.file_list[idx]
        waveform, sr = torchaudio.load(path)
        waveform = waveform.mean(dim=0)
        
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            waveform = resampler(waveform)
            
        if torch.max(torch.abs(waveform)) > 0:
            waveform = waveform / torch.max(torch.abs(waveform))
            
        if len(waveform) < self.max_samples:
            waveform = torch.nn.functional.pad(waveform, (0, self.max_samples - len(waveform)))
        else:
            waveform = waveform[:self.max_samples]
            
        return waveform, torch.tensor(label).float()