# Version-2.0/dataPipeline.py
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchaudio
from tqdm import tqdm  # <-- Crucial import for visibility

class ASVspoof5BaselineDataset(Dataset):
    """
    Standardized Dataset Loader for ASVspoof 5 tracking files.
    Handles variable-length metadata columns seamlessly with a live progress indicator.
    """
    def __init__(self, protocol_file, audio_dir, max_duration=4.0):
        self.audio_dir = audio_dir
        self.max_samples = int(16000 * max_duration)
        self.file_list = []
        self.labels = []
        
        print(f"📖 Parsing Protocol Sheet: {os.path.basename(protocol_file)}")
        
        if not os.path.exists(protocol_file):
            print(f"❌ Critical Error: Protocol file not found at '{protocol_file}'")
            return

        with open(protocol_file, 'r') as f:
            lines = f.readlines()

        print("🔍 Scanning disk for matching audio tracks...")
        # Wrapped with tqdm to provide instant feedback in the console
        for line in tqdm(lines, desc="Verifying Audio Tracks", unit="track"):
            tokens = line.strip().split()
            
            if not tokens or tokens[0] == "version" or "utterance_id" in tokens:
                continue
                
            file_id = tokens[1].strip()
            is_spoof = "spoof" in tokens
            is_bonafide = "bonafide" in tokens
            
            if not is_spoof and not is_bonafide:
                continue
                
            path = os.path.join(self.audio_dir, f"{file_id}.flac")
            
            if os.path.exists(path):
                self.file_list.append(path)
                self.labels.append(1 if is_spoof else 0)

        print(f"✅ Baseline Ingestion Ready. Active Files Found: {len(self.file_list)}")
        
        if len(self.file_list) > 0:
            num_spoofs = sum(self.labels)
            print(f"📊 Class Balance Metrics -> Spoof: {num_spoofs} | Bonafide: {len(self.file_list) - num_spoofs}")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        path = self.file_list[idx]
        label = self.labels[idx]
        
        waveform, sr = torchaudio.load(path)
        waveform = waveform.mean(dim=0)
        
        # Instance peak normalization to uniform [-1.0, 1.0] scale
        if torch.max(torch.abs(waveform)) > 0:
            waveform = waveform / torch.max(torch.abs(waveform))
        
        # ⚡ UPGRADE: Circular replication padding to eliminate edge noise artifacts
        if len(waveform) < self.max_samples:
            repeats = (self.max_samples // len(waveform)) + 1
            waveform = waveform.repeat(repeats)[:self.max_samples]
        else:
            waveform = waveform[:self.max_samples]
            
        return waveform, torch.tensor(label).float()