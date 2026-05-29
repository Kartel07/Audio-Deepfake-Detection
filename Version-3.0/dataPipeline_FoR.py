# dataPipeline_FoR.py
import os
import glob
import torch
import torchaudio
from torch.utils.data import Dataset
from tqdm import tqdm

class FakeOrReal2SecDataset(Dataset):
    def __init__(self, dataset_root_dir, split="training"):
        self.file_list = []
        self.labels = []
        self.max_samples = 32000  # 16000 Hz * 2.0s
        
        target_dir = os.path.join(dataset_root_dir, split)
        print(f"\n📖 Scanning FoR [{split.upper()}] Partition...")

        if not os.path.exists(target_dir):
            print(f"❌ Error: Path not found -> {target_dir}")
            return

        fake_paths = glob.glob(os.path.join(target_dir, "fake", "*.wav")) + glob.glob(os.path.join(target_dir, "fake", "*.flac"))
        for p in tqdm(fake_paths, desc="Parsing Synthetic Tracks"):
            self.file_list.append(p)
            self.labels.append(1) # Spoof
            
        real_paths = glob.glob(os.path.join(target_dir, "real", "*.wav")) + glob.glob(os.path.join(target_dir, "real", "*.flac"))
        for p in tqdm(real_paths, desc="Parsing Authentic Tracks"):
            self.file_list.append(p)
            self.labels.append(0) # Bona-fide

        print(f"✅ Loaded: {len(self.file_list)} Tracks | Balance -> Spoof: {sum(self.labels)} | Bonafide: {len(self.file_list) - sum(self.labels)}")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        path = self.file_list[idx]
        label = self.labels[idx]
        
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