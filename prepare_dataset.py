import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from mel_spec import process_chunk, save_spectrogram
import random

def get_wav_files(dataset_root, train_ratio=0.8):
    """Recursively collects all wav files and splits them into training and validation sets."""
    dataset_root = Path(dataset_root)
    file_dict = {"music": [], "noise": [], "speech": []}
    
    for category in file_dict.keys():
        for wav_path in dataset_root.rglob(f"*/{category}/*.wav"):
            file_dict[category].append(wav_path)
    
    train_files, valid_files = {}, {}
    for category, files in file_dict.items():
        random.shuffle(files)
        split_idx = int(len(files) * train_ratio)
        train_files[category] = files[:split_idx]
        valid_files[category] = files[split_idx:]
    
    return train_files, valid_files
    

def preprocess_audio(file_list, output_folder, chunk_size=16*1024):
    """Converts wav files into mel spectrogram images."""
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    for category, files in file_list.items():
        output_category_path = output_folder / category
        output_category_path.mkdir(parents=True, exist_ok=True)
        
        for file_no, wav_file in enumerate(files[1:10000]):
            print(category, file_no)
            signal, sr = librosa.load(wav_file)
            num_chunks = len(signal) // chunk_size
            
            for i in range(num_chunks):
                chunk = signal[i * chunk_size:(i + 1) * chunk_size]
                spectrogram_data = process_chunk(chunk, sr)
                chunk_filename = output_category_path / f"{wav_file.stem}_{i}.png"
                save_spectrogram(spectrogram_data, chunk_filename)

# Paths
dataset_root = os.path.expanduser('~') + "/.cache/kagglehub/datasets/snirjhar/audioset-speech-music-noise-4k/versions/4"
train_img_path = "processed/training"
valid_img_path = "processed/validation"

dataset_path = "processed"

# Collect and split files
train_files, valid_files = get_wav_files(dataset_root)

# Preprocessing
preprocess_audio(train_files, train_img_path)
preprocess_audio(valid_files, valid_img_path)