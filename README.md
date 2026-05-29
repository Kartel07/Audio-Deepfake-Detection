# Audio-Deepfake-Detection
A cross-corpus evaluation framework analyzing the impact of architectural inductive biases (2D CNN, ResNet, 1D RawNet2, and self-supervised XLS-R Transformers) on speech deepfake detection under acoustic domain shift from Fake-or-Real (FoR) to ASVspoof 5.


# Empirical Evaluation of Architectural Inductive Biases in Audio Deepfake Detection Under Acoustic Domain Shift

This repository contains the complete implementation, evaluation pipelines, and raw benchmark score exports for a controlled comparative study exploring how distinct neural network topologies generalize under severe cross-corpus acoustic domain shift. 

The framework isolates model architecture as the primary independent variable by training across five system topologies using a highly constrained source environment and evaluating zero-shot performance against an unconstrained, crowdsourced target environment.

## Project Overview

Current state-of-the-art automatic speaker verification and deepfake countermeasures achieve near-perfect metrics under in-domain evaluations. However, these paradigms are susceptible to shortcut learning—overfitting to localized spectral artifacts, silence padding patterns, or stationary microphone impulse responses unique to the training distribution. 

This project establishes a standardized benchmark to analyze how different modeling approaches handle severe acoustic shifts.

### Evaluated Topologies
* **Version 1 (CNN Baseline):** A standard 2D Convolutional Neural Network processing log-mel spectrograms to target localized spatial-acoustic anomalies.
* **Version 2 (Deep ResNet):** A deep Residual Network designed to evaluate whether scaling hierarchical spatial capacity improves representation invariance.
* **Version 3 (XLS-R Transformer):** A 1D raw-waveform backend utilizing a self-supervised, 300-million parameter multilingual XLS-R architecture operating via global self-attention.
* **Version 4 (RawNet2):** A specialized speech forensic architecture integrating a learnable 1D SincNet frontend to enforce physical band-pass filter constraints directly on time-domain signals.
* **Version 5 (Orthogonal Fusion Engine):** A late-stage late-fusion framework aggregating unweighted posterior probabilities from the deep ResNet and the self-supervised Transformer to exploit decoupled error surfaces.

---

## Experimental Protocol

### Datasets
1. **Source Domain (Training):** Fake-or-Real (FoR) dataset. Characterized by clean acoustic backgrounds and symmetrical distribution profiles.
2. **Target Domain (Zero-Shot Testing):** ASVspoof 5 Developmental Track. Highly variable acoustic profiles featuring diverse background noise conditions, compression artifacts, and unseen generative algorithms.

### Standardization Pipeline
All audio data undergoes deterministic preprocessing prior to model feature engineering:
* Monophonic channel reduction and strict resampling to $16\text{ kHz}$.
* Temporal alignment to a rigid 4.0-second window ($64,000$ discrete samples) via symmetric zero-padding or uniform truncation.
* Log-mel spectrograms computed using a 1024-point FFT, 256-sample hop length, and 128 Mel bins (where applicable).

---

## Empirical Results

The models exhibit starkly divergent vulnerability patterns when transitioned from the source to the target environment. Performance is measured via Equal Error Rate (EER), where the False Acceptance Rate (FAR) intersects the False Rejection Rate (FRR).

| Architecture | Input Domain | In-Domain EER | Cross-Corpus EER | $\Delta$ EER |
| :--- | :--- | :---: | :---: | :---: |
| **V1 (CNN Baseline)** | 2D Spectrogram | 8.50% | 58.85% | +50.35% |
| **V2 (Deep ResNet)** | 2D Spectrogram | 4.10% | 67.48% | +63.38% |
| **V4 (RawNet2)** | 1D Raw Waveform | 3.40% | 55.66% | +52.26% |
| **V3 (XLS-R)** | 1D Raw Waveform | 2.80% | 18.37% | +15.57% |
| **V5 (Orthogonal Fusion)**| Mixed (1D + 2D) | — | **12.15%** | — |

### Key Insights
* **Spatial and Physical Overfitting:** Purely convolutional spatial layers (V1, V2) and parameter-optimized physical frontends (V4) collapse under dataset shift, showing extreme sensitivity to out-of-domain channel acoustics.
* **Self-Supervised Stability:** Large-scale multilingual pre-training (V3) provides superior domain abstraction, anchoring features in semantic vocal tract characteristics rather than acoustic background conditions.
* **Error Decorrelation:** Merging disjoint 1D and 2D modeling topologies stabilizes post-classification decision vectors, mitigating individual failure loops.

---

## Reproducibility and Hardware Specifications

To support complete benchmark verifiability, all training and evaluation passes were locked to a single workstation execution profile:

* **Hardware Configuration:** Single NVIDIA GeForce RTX 3050 Laptop GPU (6 GB VRAM); 13th Gen Intel Core i7-13650HX CPU (14 cores, 20 logical processors, 2.60 GHz base frequency).
* **Deterministic Configuration:** Fixed seed initializations executed globally across NumPy, PyTorch, and back-end CUDA engines. Deterministic cuDNN execution was strictly enforced.
* **Memory Management:** High-parameter model processing (V3 and V5) was mapped to a reduced batch size of 16 and stabilized via PyTorch Automatic Mixed Precision (AMP) utilizing `torch.amp.autocast`.

### Core Software Environment
* Python 3.10.11
* CUDA 12.4
* PyTorch 2.6.0+cu124
* Torchaudio 2.6.0+cu124
* Transformers 5.9.0

---

## Installation and Execution

### 1. Environment Setup
Clone the repository and build the workspace environment dependencies:
```bash
git clone [https://github.com/yourusername/audio-deepfake-domain-shift.git](https://github.com/yourusername/audio-deepfake-domain-shift.git)
cd audio-deepfake-domain-shift
conda env create -f environment.yml
conda activate deepfake-env
