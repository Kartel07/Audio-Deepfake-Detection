import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def plot_spectrogram(audio_path, title, ax):
    y, sr = librosa.load(audio_path, sr=16000, duration=4.0)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    S_dB = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000, ax=ax)
    ax.set_title(title, fontsize=14, fontweight='bold')
    return img

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# ADDED the 'r' prefix to make these raw strings, and matched the paths to the correct titles
plot_spectrogram(r"D:\MYFILES\kartik\CS_Projects\DeepFake Audio System\Datasets\for-2sec\testing\real\file3.wav_16k.wav_norm.wav_mono.wav_silence.wav_2sec.wav", "Authentic Speech (Log-Mel)", axes[0])

img = plot_spectrogram(r"D:\MYFILES\kartik\CS_Projects\DeepFake Audio System\Datasets\for-2sec\testing\fake\file2.wav_16k.wav_norm.wav_mono.wav_silence.wav_2sec.wav", "Deepfake Speech (Log-Mel)", axes[1])

fig.colorbar(img, ax=axes, format='%+2.0f dB')
plt.suptitle("Version 1.0 Input: Audio as an Image", fontsize=16, fontweight='bold', y=1.05)
plt.savefig("v1_spectrograms.png", dpi=300, bbox_inches='tight')
plt.show()