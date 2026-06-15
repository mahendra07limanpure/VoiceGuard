import random
import numpy as np
import librosa
from tqdm import tqdm
from utils.feature_extraction import extract_features_from_audio

def augment_audio(audio, sr=16000):
    """
    Apply random augmentation to audio array.
    Randomly pick ONE augmentation per call.
    
    Options:
    1. Pitch shift: librosa.effects.pitch_shift(audio, sr, n_steps=random.choice([-2,-1,1,2]))
    2. Time stretch: librosa.effects.time_stretch(audio, rate=random.choice([0.9, 1.1]))
    3. Gaussian noise: audio + np.random.normal(0, 0.003, len(audio))
    4. Time masking (SpecAugment style):
       - Pick random start index and mask 10-20% of audio with zeros
    
    Returns: augmented audio array
    """
    # Ensure audio is float32
    audio = audio.astype(np.float32)
    
    # Randomly select one of the four augmentation options
    opt = random.choice([1, 2, 3, 4])
    
    try:
        if opt == 1:
            # Pitch shift: ±1 or ±2 semitones
            n_steps = random.choice([-2, -1, 1, 2])
            return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
            
        elif opt == 2:
            # Time stretch: 0.9x or 1.1x
            rate = random.choice([0.9, 1.1])
            return librosa.effects.time_stretch(audio, rate=rate)
            
        elif opt == 3:
            # Gaussian noise: mean 0, std 0.003
            noise = np.random.normal(0, 0.003, len(audio))
            return audio + noise
            
        else:
            # Time masking: mask a random 10-20% segment of audio with zeros
            length = len(audio)
            mask_pct = random.uniform(0.10, 0.20)
            mask_len = int(length * mask_pct)
            if length > mask_len:
                start_idx = random.randint(0, length - mask_len)
                augmented = audio.copy()
                augmented[start_idx : start_idx + mask_len] = 0
                return augmented
            return audio
    except Exception as e:
        # In case of any error, return the original audio unchanged
        return audio

def augment_dataset(X, y, augment_factor=2):
    """
    Double or multiply the dataset size using augmentation.
    For each sample in X (a list of file paths or raw audio arrays), 
    create augment_factor new versions.
    Apply augmentation at audio level before feature extraction.
    Returns: X_aug, y_aug (concatenated with originals)
    """
    X_aug_list = []
    y_aug_list = []
    sr = 16000
    
    print(f"Starting dataset augmentation (augment_factor={augment_factor})...")
    
    # To force raw features extraction in augment_dataset, we pass mean=[0,0] and std=[1,1]
    raw_mean = [0.0, 0.0]
    raw_std = [1.0, 1.0]
    
    for item, label in tqdm(zip(X, y), total=len(X), desc="Augmenting dataset"):
        try:
            # 1. Load/get original audio
            if isinstance(item, str):
                y_audio, _ = librosa.load(item, sr=sr, mono=True)
            else:
                y_audio = item.astype(np.float32)
                
            # 2. Extract raw features from original audio
            feat_orig = extract_features_from_audio(y_audio, sr=sr, mean=raw_mean, std=raw_std)
            if feat_orig is not None:
                X_aug_list.append(feat_orig)
                y_aug_list.append(label)
                
            # 3. Create augment_factor augmented versions
            for _ in range(augment_factor):
                y_aug = augment_audio(y_audio, sr=sr)
                feat_aug = extract_features_from_audio(y_aug, sr=sr, mean=raw_mean, std=raw_std)
                if feat_aug is not None:
                    X_aug_list.append(feat_aug)
                    y_aug_list.append(label)
        except Exception as e:
            continue
            
    X_aug = np.array(X_aug_list, dtype=np.float32)
    y_aug = np.array(y_aug_list, dtype=np.int32)
    
    print(f"Augmentation complete. Final shape: {X_aug.shape}")
    return X_aug, y_aug
