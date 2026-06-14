import os
import pickle
import numpy as np
import tensorflow as tf
from utils.feature_extraction import extract_features

def predict_single(file_path, 
                   model_path='model/voiceguard_cnn_lstm.h5',
                   norm_path='model/norm_params.pkl',
                   threshold_path='model/threshold.pkl'):
    """
    Full inference pipeline for a single audio file.
    
    Steps:
    1. Load audio
    2. Extract MFCC + Mel Spectrogram features
    3. Normalize using saved mean/std
    4. Reshape to (1, 128, 128, 2)
    5. Model predict → raw probability
    6. Apply optimal threshold
    7. Return dict:
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
        
    # Load normalization parameters if available
    mean, std = None, None
    if os.path.exists(norm_path):
        with open(norm_path, 'rb') as f:
            norm_params = pickle.load(f)
            mean = norm_params.get('mean')
            std = norm_params.get('std')
            
    # Extract features (performs self-normalization first)
    features = extract_features(file_path)
    if features is None:
        raise ValueError(f"Could not extract features from {file_path}. File may be corrupted.")
        
    # Apply global normalization using saved mean and std
    if mean is not None and std is not None:
        features[:, :, 0] = (features[:, :, 0] - mean[0]) / (std[0] + 1e-8)
        features[:, :, 1] = (features[:, :, 1] - mean[1]) / (std[1] + 1e-8)
        
    # Reshape to (1, 128, 128, 2) for model input
    features_input = np.expand_dims(features, axis=0)
    
    # Load model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Train the model first.")
        
    # Load TensorFlow Keras model
    model = tf.keras.models.load_model(model_path)
    
    # Predict raw probability
    prob = float(model.predict(features_input, verbose=0)[0][0])
    
    # Load optimal threshold
    threshold = 0.5
    if os.path.exists(threshold_path):
        with open(threshold_path, 'rb') as f:
            threshold = pickle.load(f)
            
    # Determine label and confidence based on threshold
    if prob >= threshold:
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
    parser.add_argument('--model', default='model/voiceguard_cnn_lstm.h5', help="Path to model file")
    parser.add_argument('--norm', default='model/norm_params.pkl', help="Path to normalization parameters")
    parser.add_argument('--threshold', default='model/threshold.pkl', help="Path to threshold parameter")
    args = parser.parse_args()
    
    try:
        result = predict_single(
            args.audio,
            model_path=args.model,
            norm_path=args.norm,
            threshold_path=args.threshold
        )
        print(f"\nFile         : {args.audio}")
        print(f"Result       : {result['label']}")
        print(f"Confidence   : {result['confidence']:.2f}%")
        print(f"Fake Prob    : {result['fake_probability']:.4f}")
        print(f"Real Prob    : {result['real_probability']:.4f}")
        print(f"Threshold    : {result['threshold_used']:.4f}")
    except Exception as e:
        print(f"Error during prediction: {e}")
