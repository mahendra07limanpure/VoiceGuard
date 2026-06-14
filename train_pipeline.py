import os
import argparse
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import librosa
import librosa.display
import pickle

from utils.feature_extraction import extract_features, extract_features_from_audio, load_dataset
from utils.augmentation import augment_dataset
from utils.metrics import calculate_metrics, save_norm_params

def build_model(input_shape=(128, 128, 2)):
    inputs = tf.keras.layers.Input(shape=input_shape)
    
    # CNN BLOCK
    x = tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2,2))(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    x = tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2,2))(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    x = tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2,4))(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    # RESHAPE for LSTM
    x = tf.keras.layers.Reshape((8, -1))(x)
    
    # BiLSTM BLOCK
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=False))(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    # Output
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    return model

def main():
    parser = argparse.ArgumentParser(description="VoiceGuard Training Pipeline")
    parser.add_argument('--limit', type=int, default=1500, help="Limit files per class for training speed")
    parser.add_argument('--epochs', type=int, default=15, help="Number of epochs to train")
    args = parser.parse_args()
    
    # Setup directories
    os.makedirs('model', exist_ok=True)
    os.makedirs('assets', exist_ok=True)
    os.makedirs('features', exist_ok=True)
    
    # Paths
    train_dir = 'data/for-2seconds/training'
    test_dir = 'data/for-2seconds/testing'
    
    # 1. Collect file lists
    categories = {'real': 0, 'fake': 1}
    train_files = []
    train_labels = []
    
    for cat_name, cat_label in categories.items():
        cat_dir = os.path.join(train_dir, cat_name)
        files = [os.path.join(cat_dir, f) for f in os.listdir(cat_dir) if f.lower().endswith(('.wav', '.mp3', '.flac'))]
        # Apply limit to keep train time reasonable on CPU
        if len(files) > args.limit:
            import random
            random.seed(42)
            files = random.sample(files, args.limit)
        train_files.extend(files)
        train_labels.extend([cat_label] * len(files))
        
    print(f"Collected {len(train_files)} files for training split.")
    
    # 2. Data Augmentation
    # Double the dataset size: original + 1 augmented version per sample
    X_train, y_train = augment_dataset(train_files, train_labels, augment_factor=1)
    
    # Compute and save normalization params from training features
    save_norm_params(X_train)
    
    # 3. Load Testing data (no augmentation)
    print("Extracting testing features...")
    X_test, y_test = load_dataset(test_dir, save_path='features')
    
    # 4. Compute class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(enumerate(class_weights))
    print(f"Class weights: {class_weight_dict}")
    
    # 5. Build Model
    model = build_model()
    model.summary()
    
    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=7, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3),
        tf.keras.callbacks.ModelCheckpoint('model/voiceguard_cnn_lstm.h5', save_best_only=True)
    ]
    
    # 6. Train
    print("Starting training...")
    history = model.fit(
        X_train, y_train,
        epochs=args.epochs,
        batch_size=32,
        validation_split=0.2,
        class_weight=class_weight_dict,
        callbacks=callbacks
    )
    
    # Save training curves
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(history.history['loss'], label='Train Loss', color='#00C9A7', linewidth=2)
    if 'val_loss' in history.history:
        ax1.plot(history.history['val_loss'], label='Val Loss', color='#ff7b72', linewidth=2)
    ax1.set_title('Model Loss', fontsize=12)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.1)
    
    ax2.plot(history.history['accuracy'], label='Train Acc', color='#00C9A7', linewidth=2)
    if 'val_accuracy' in history.history:
        ax2.plot(history.history['val_accuracy'], label='Val Acc', color='#ff7b72', linewidth=2)
    ax2.set_title('Model Accuracy', fontsize=12)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.1)
    
    plt.tight_layout()
    plt.savefig('assets/training_curves.png', dpi=300)
    plt.close()
    print("Saved training curves to assets/training_curves.png")
    
    # 7. Evaluate
    print("Evaluating model...")
    y_prob = model.predict(X_test, batch_size=32)
    y_prob = y_prob.flatten()
    
    metrics = calculate_metrics(y_test, y_prob, save_dir='assets', model_dir='model')
    
    # 8. Save sample spectrogram for assets
    print("Generating sample spectrogram...")
    sample_file = train_files[0]
    y_audio, sr = librosa.load(sample_file, sr=16000, mono=True)
    mel_spec = librosa.feature.melspectrogram(y=y_audio, sr=16000, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    plt.figure(figsize=(8, 4))
    img = librosa.display.specshow(mel_spec_db, x_axis='time', y_axis='mel', sr=16000, cmap='magma')
    plt.title('Sample Mel Spectrogram', color='#E6EDF3')
    plt.colorbar(img, format='%+2.0f dB')
    plt.tight_layout()
    plt.savefig('assets/sample_spectrogram.png', dpi=300)
    plt.close()
    print("Saved sample spectrogram to assets/sample_spectrogram.png")

if __name__ == '__main__':
    main()
