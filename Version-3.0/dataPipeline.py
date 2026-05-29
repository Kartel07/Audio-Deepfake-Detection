# Version-3.0/dataPipeline.py
import os
import torch
from torch.utils.data import Dataset
import torchaudio
from tqdm import tqdm

class ASVspoof5TransformerDataset(Dataset):
    """
    Advanced Transformer-ready Data Ingestion Engine for ASVspoof 5.
    Generates explicit temporal attention masks for sequence operations.
    """
    def __init__(self, protocol_file, audio_dir, max_duration=4.0):
        self.audio_dir = audio_dir
        self.max_samples = int(16000 * max_duration)  # 64,000 samples for 4 seconds
        self.file_list = []
        self.labels = []
        
        print(f"📖 Parsing Transformer Protocol Sheet: {os.path.basename(protocol_file)}")
        with open(protocol_file, 'r') as f:
            lines = f.readlines()

        print("🔍 Extracting file mappings and building verification tree...")
        for line in tqdm(lines, desc="Validating Tracks on Disk", unit="track"):
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

        print(f"✅ Ingestion Complete. Found {len(self.file_list)} active verified tracks.")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        path = self.file_list[idx]
        label = self.labels[idx]
        
        waveform, sr = torchaudio.load(path)
        waveform = waveform.mean(dim=0)  # Convert to mono
        
        # Create binary attention mask sequence (1 = valid audio, 0 = padding)
        audio_len = len(waveform)
        if audio_len < self.max_samples:
            pad_len = self.max_samples - audio_len
            waveform = torch.nn.functional.pad(waveform, (0, pad_len))
            attention_mask = torch.cat([torch.ones(audio_len), torch.zeros(pad_len)])
        else:
            waveform = waveform[:self.max_samples]
            attention_mask = torch.ones(self.max_samples)
            
        return waveform, attention_mask, torch.tensor(label).float()