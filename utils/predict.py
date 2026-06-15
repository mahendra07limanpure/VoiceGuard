import os
import json
import numpy as np
import tensorflow as tf
import librosa
from scipy.fftpack import dct

# ── Constants (must match the training configuration of best_model.keras) ────────────────
SR            = 16000
DURATION      = 4.0
TARGET_FRAMES = 128
N_LFCC        = 60
N_MFCC        = 60
N_FFT         = 512
HOP_LENGTH    = 160
N_FILTERS     = 128

def compute_lfcc(audio):
    """Linear Frequency Cepstral Coefficients — anti-spoofing standard."""
    stft  = np.abs(librosa.stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH)) ** 2
    freqs = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)
    lin_f = np.linspace(0, SR // 2, N_FILTERS + 2)
    fb    = np.zeros((N_FILTERS, len(freqs)))
    for m in range(1, N_FILTERS + 1):
        fl, fc, fr = lin_f[m-1], lin_f[m], lin_f[m+1]
        for k, f in enumerate(freqs):
            if fl <= f <= fc:  fb[m-1, k] = (f  - fl) /(fc - fl + 1e-8)
            elif fc < f <= fr: fb[m-1, k] = (fr - f) /(fr - fc + 1e-8)
    log_spec = np.log(np.dot(fb, stft) + 1e-8)
    return dct(log_spec, type=2, axis=0, norm='ortho')[:N_LFCC]

def extract_features(file_path):
    """Extract LFCC + MFCC + Delta-LFCC features. Returns shape (1, 180, 128, 1)."""
    n = int(SR * DURATION)
    try:
        audio, _ = librosa.load(file_path, sr=SR, duration=DURATION)
    except Exception as e:
        raise ValueError(f"Could not load audio file: {e}")

    audio = np.pad(audio, (0, max(0, n - len(audio))))[:n]
    # Pre-emphasis filter
    audio = np.append(audio[0], audio[1:] - 0.97 * audio[:-1]).astype(np.float32)

    lfcc       = compute_lfcc(audio)
    mfcc       = librosa.feature.mfcc(y=audio, sr=SR, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
    lfcc_delta = librosa.feature.delta(lfcc)
    features   = np.concatenate([lfcc, mfcc, lfcc_delta], axis=0)

    T = features.shape[1]
    features = np.pad(features, ((0, 0), (0, max(0, TARGET_FRAMES - T))))[:, :TARGET_FRAMES]

    # Per-sample normalization
    mean = features.mean(axis=1, keepdims=True)
    std  = features.std(axis=1, keepdims=True) + 1e-6
    features = (features - mean) / std

    return features.astype(np.float32)[np.newaxis, ..., np.newaxis]  # (1, 180, 128, 1)

def predict_single(file_path, 
                   model_path='model/best_model.keras',
                   config_path='model/model_config.json',
                   **kwargs):
    """
    Full inference pipeline for a single audio file matching the new model configuration.
    
    Returns a dict with verification details:
       {
         'label': 'Genuine (Human)' or 'Deepfake (AI-Generated)',
         'confidence': float (0-100),
         'fake_probability': float,
         'real_probability': float,
         'threshold_used': float
       }
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found at: {file_path}")
        
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
        
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
        
    # Extract features
    features_input = extract_features(file_path)
    
    # Load model and config
    model = tf.keras.models.load_model(model_path)
    with open(config_path) as f:
        cfg = json.load(f)
    threshold = cfg.get("threshold", 0.05)
    
    # Run prediction
    prob = float(model.predict(features_input, verbose=0)[0][0])
    
    # Determine label and confidence based on threshold
    if prob > threshold:
        label = 'Deepfake (AI-Generated)'
        # Scale probability [threshold, 1.0] -> [50, 100] % confidence
        if (1.0 - threshold) > 0:
            confidence = 50.0 + ((prob - threshold) / (1.0 - threshold)) * 50.0
        else:
            confidence = 100.0
    else:
        label = 'Genuine (Human)'
        # Scale probability [0, threshold] -> [100, 50] % confidence
        if threshold > 0:
            confidence = 50.0 + ((threshold - prob) / threshold) * 50.0
        else:
            confidence = 100.0
            
    # Clip confidence to [0.0, 100.0]
    confidence = min(max(confidence, 0.0), 100.0)
            
    return {
        'label': label,
        'confidence': confidence,
        'fake_probability': prob,
        'real_probability': 1.0 - prob,
        'threshold_used': threshold
    }

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="VoiceGuard Single Audio Inference")
    parser.add_argument('--audio', required=True, help="Path to the audio file to analyze")
    parser.add_argument('--model', default='model/best_model.keras', help="Path to model file")
    parser.add_argument('--config', default='model/model_config.json', help="Path to config file")
    args = parser.parse_args()
    
    try:
        result = predict_single(
            args.audio,
            model_path=args.model,
            config_path=args.config
        )
        print(f"\n{'='*45}")
        print("  VoiceGuard Prediction Result")
        print(f"{'='*45}")
        print(f"File         : {args.audio}")
        print(f"Result       : {result['label']}")
        print(f"Confidence   : {result['confidence']:.2f}%")
        print(f"Fake Prob    : {result['fake_probability']:.4f}")
        print(f"Real Prob    : {result['real_probability']:.4f}")
        print(f"Threshold    : {result['threshold_used']:.4f}")
        print(f"{'='*45}\n")
    except Exception as e:
        print(f"Error during prediction: {e}")
