import os
import numpy as np
import librosa
from tqdm import tqdm
from scipy.interpolate import interp1d

def extract_features_from_audio(y, sr=16000, n_mfcc=40, n_mels=128, max_frames=128):
    """
    Helper function to extract dual-channel feature from pre-loaded/augmented audio array.
    """
    try:
        # Guard for empty file
        if len(y) == 0:
            return None
            
        # Normalize amplitude to [-1, 1]
        max_val = np.max(np.abs(y))
        if max_val > 0:
            y = y / max_val
        else:
            return None
            
        # Extract MFCC: shape (40, T)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        
        # Extract Mel Spectrogram: shape (128, T)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        # Convert to log scale (dB)
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Interpolate MFCC height from 40 to 128
        T = mfcc.shape[1]
        x = np.linspace(0, 1, n_mfcc)
        x_new = np.linspace(0, 1, n_mels)
        f = interp1d(x, mfcc, axis=0, kind='linear', fill_value="extrapolate")
        mfcc_resized = f(x_new)
        
        # Pad or truncate on time axis to max_frames=128
        if T < max_frames:
            pad_width = max_frames - T
            mfcc_padded = np.pad(mfcc_resized, ((0, 0), (0, pad_width)), mode='constant')
            mel_padded = np.pad(mel_spec, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc_padded = mfcc_resized[:, :max_frames]
            mel_padded = mel_spec[:, :max_frames]
            
        # Stack as shape (128, 128, 2)
        features = np.stack([mfcc_padded, mel_padded], axis=-1)
        
        # Normalize each channel independently (zero mean, unit std)
        m0 = np.mean(features[:, :, 0])
        s0 = np.std(features[:, :, 0])
        features[:, :, 0] = (features[:, :, 0] - m0) / (s0 + 1e-8)
        
        m1 = np.mean(features[:, :, 1])
        s1 = np.std(features[:, :, 1])
        features[:, :, 1] = (features[:, :, 1] - m1) / (s1 + 1e-8)
        
        return features
    except Exception as e:
        return None

def extract_features(file_path, sr=16000, n_mfcc=40, 
                     n_mels=128, max_frames=128):
    """
    Extract dual-channel feature: MFCC + Mel Spectrogram from a file path.
    """
    try:
        y, sample_rate = librosa.load(file_path, sr=sr, mono=True)
        return extract_features_from_audio(y, sr=sr, n_mfcc=n_mfcc, n_mels=n_mels, max_frames=max_frames)
    except Exception as e:
        return None

def load_dataset(data_dir, save_path='features'):
    """
    Load all audio files, extract features, save as .npy
    
    - Walk through real/ and fake/ subdirectories
    - Show tqdm progress bar
    - Skip None returns (corrupted files)
    - Print class distribution
    - Save: X_train.npy (N, 128, 128, 2), y_train.npy (N,) (or appropriate prefix)
    - Return X, y arrays
    """
    os.makedirs(save_path, exist_ok=True)
    
    # Check split based on data_dir name
    dir_name = os.path.basename(os.path.normpath(data_dir))
    if 'train' in dir_name.lower():
        prefix = 'train'
    elif 'val' in dir_name.lower():
        prefix = 'val'
    elif 'test' in dir_name.lower():
        prefix = 'test'
    else:
        prefix = dir_name
        
    X_list = []
    y_list = []
    
    categories = {'real': 0, 'fake': 1}
    
    # Collect all audio files and their labels
    file_list = []
    for cat_name, cat_label in categories.items():
        cat_dir = os.path.join(data_dir, cat_name)
        if not os.path.exists(cat_dir):
            continue
        for root, _, files in os.walk(cat_dir):
            for file in files:
                if file.lower().endswith(('.wav', '.mp3', '.flac', '.m4a')):
                    file_list.append((os.path.join(root, file), cat_label))
                    
    print(f"Found {len(file_list)} files in {data_dir}. Starting feature extraction...")
    
    # Extract features with tqdm progress bar
    for file_path, label in tqdm(file_list, desc=f"Extracting {prefix} features"):
        feat = extract_features(file_path)
        if feat is not None:
            X_list.append(feat)
            y_list.append(label)
            
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    
    # Print class distribution
    unique, counts = np.unique(y, return_counts=True)
    dist = dict(zip(unique, counts))
    print(f"\nClass distribution for {prefix}:")
    print(f"  Real (0): {dist.get(0, 0)}")
    print(f"  Fake (1): {dist.get(1, 0)}")
    
    # Save files
    x_save_file = os.path.join(save_path, f"X_{prefix}.npy")
    y_save_file = os.path.join(save_path, f"y_{prefix}.npy")
    np.save(x_save_file, X)
    np.save(y_save_file, y)
    print(f"Saved: {x_save_file} (shape: {X.shape}), {y_save_file} (shape: {y.shape})")
    
    return X, y
