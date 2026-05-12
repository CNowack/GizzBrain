# gizzbrain/encoder.py

import os
import librosa
import numpy as np
import pandas as pd
import math
import torch
from torch.utils.data import Dataset

def create_chunk_index(df_original, chunk_length=5.0, trim_seconds=30.0):
    """
    Takes a dataframe of whole songs and creates a new dataframe
    where every row represents a specific time-chunk of a song.

    trim_seconds: chunks that start or end within this many seconds of the
    file boundaries are excluded to avoid applause/crowd noise.
    """
    chunk_data = []

    for _, row in df_original.iterrows():
        path = row['path']
        try:
            # Get exact length of the MP3 without loading the heavy audio data
            total_duration = librosa.get_duration(path=path)
            num_chunks = math.floor(total_duration / chunk_length)

            for i in range(num_chunks):
                start_time = i * chunk_length
                end_time = start_time + chunk_length
                if start_time < trim_seconds or end_time > total_duration - trim_seconds:
                    continue
                chunk_data.append({
                    'path': path,
                    'title': row['title'],
                    'start_time': start_time,
                    'duration': chunk_length
                })
        except Exception as e:
            print(f"Failed to index {path}: {e}")

    return pd.DataFrame(chunk_data)

def audio_to_tensor(path, offset=0.0, duration=5.0, sr=22050, n_mels=128):
    """
    Loads a specific chunk of an audio file and converts it to a PyTorch tensor.
    """
    # Load ONLY the requested time chunk
    y, sr = librosa.load(path, sr=sr, offset=offset, duration=duration)
    
    # Strict Length Enforcement (Critical for CNNs to have identical tensor sizes)
    target_length = int(sr * duration)
    if len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y)))
    elif len(y) > target_length:
        y = y[:target_length]

    # Create Mel Spectrogram
    spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    spec_db = librosa.power_to_db(spec, ref=np.max)
    
    # Convert to Tensor [Channel, Height, Width] -> [1, 128, time_steps]
    tensor = torch.from_numpy(spec_db).float().unsqueeze(0)
    
    return tensor

def precompute_chunks(chunk_df, output_dir, sr=22050, n_mels=128):
    """
    Convert every chunk in chunk_df to a .pt tensor file saved in output_dir.
    Returns chunk_df with a 'tensor_path' column added.
    Run this once before training to avoid recomputing spectrograms every epoch.
    """
    from tqdm import tqdm
    os.makedirs(output_dir, exist_ok=True)

    tensor_paths = []
    for i, (_, row) in enumerate(tqdm(chunk_df.iterrows(), total=len(chunk_df), desc="Pre-computing spectrograms")):
        tensor = audio_to_tensor(row['path'], offset=row['start_time'], duration=row['duration'], sr=sr, n_mels=n_mels)
        out_path = os.path.join(output_dir, f"{i:07d}.pt")
        torch.save(tensor, out_path)
        tensor_paths.append(out_path)

    result = chunk_df.copy().reset_index(drop=True)
    result['tensor_path'] = tensor_paths
    return result


class GizzDataset(Dataset):
    """
    A PyTorch Dataset that loads audio chunks "just-in-time" for the neural network.
    """
    def __init__(self, chunk_df, label_mapping=None, sr=22050, n_mels=128):
        self.chunk_df = chunk_df
        self.sr = sr
        self.n_mels = n_mels
        
        # Create a numerical label column for PyTorch if not provided
        if label_mapping is None:
            self.chunk_df['label'] = self.chunk_df['title'].astype('category').cat.codes
        else:
            self.chunk_df['label'] = self.chunk_df['title'].map(label_mapping)

    def __len__(self):
        return len(self.chunk_df)

    def __getitem__(self, idx):
        row = self.chunk_df.iloc[idx]

        if 'tensor_path' in self.chunk_df.columns:
            tensor = torch.load(row['tensor_path'], weights_only=True)
        else:
            tensor = audio_to_tensor(
                path=row['path'],
                offset=row['start_time'],
                duration=row['duration'],
                sr=self.sr,
                n_mels=self.n_mels
            )

        label = torch.tensor(row['label'], dtype=torch.long)
        return tensor, label