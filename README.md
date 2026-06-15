# 🎙️ VoiceGuard
### Detect the Deception, Trust the Voice

VoiceGuard is a high-performance, original deepfake audio detection system built for the MARS Open Projects 2026. It utilizes a novel **spatial-temporal hybrid learning architecture (CNN + Bidirectional LSTM)** to identify AI-synthesized audio, speaker spoofing, and synthetic voice anomalies.

---

## 🚨 The Problem
AI voice cloning and deepfakes have progressed to a point where they are nearly indistinguishable from human voices to the human ear. Traditional audio verification systems often fail when presented with sophisticated temporal or spectral manipulations. VoiceGuard solves this by extracting linear-frequency cepstral coefficients (LFCC), Mel-frequency cepstral coefficients (MFCC), and Delta-LFCC features, stacking them as a spatial input tensor, and processing them with a hybrid CNN-BiLSTM network to capture both spatial frequency anomalies and temporal discontinuities.

---

## ✅ Results — All 5 Criteria
VoiceGuard is evaluated on the test set of the Fake-or-Real (FoR) dataset, achieving outstanding results that comfortably surpass all evaluation thresholds:

| Metric | Target | VoiceGuard Result | Status |
| :--- | :--- | :--- | :--- |
| **Overall Accuracy** | ≥ 80% | **89.56%** | PASS |
| **F1 Score** | ≥ 80% | **89.12%** | PASS |
| **EER (Equal Error Rate)** | ≤ 12% | **6.24%** | PASS |
| **Genuine Accuracy** | ≥ 75% | **98.24%** | PASS |
| **Deepfake Accuracy** | ≥ 75% | **81.28%** | PASS |

* **ROC AUC**: **96.84%**
* **Decision Threshold**: **0.05** (optimized for balanced per-class accuracy on validation set)

---

## 🧠 How It Works
The processing pipeline extracts cepstral patterns and models them sequentially:

```text
┌───────────────────────────────────────────────────────────┐
│               Input Audio (.wav, .mp3, .flac)             │
└─────────────────────────────┬─────────────────────────────┘
                              ▼
┌───────────────────────────────────────────────────────────┐
│  LFCC (60 x T)  │  MFCC (60 x T)  │  Delta-LFCC (60 x T)  │
└─────────┬───────┴────────┬────────┴───────────┬───────────┘
          └────────────────┼────────────────────┘
                           ▼ (Concatenate & Normalize)
┌───────────────────────────────────────────────────────────┐
│           Spatial Feature Tensor (180 x 128 x 1)          │
└─────────────────────────────┬─────────────────────────────┘
                              ▼
┌───────────────────────────────────────────────────────────┐
│       4-Block CNN (Conv2D -> BN -> Swish -> MaxPool)      │
│       Outputs spatial/frequency features (15x8x128)       │
└─────────────────────────────┬─────────────────────────────┘
                              ▼
┌───────────────────────────────────────────────────────────┐
│            Reshape to Sequential Form (15, 1024)          │
└─────────────────────────────┬─────────────────────────────┘
                              ▼
┌───────────────────────────────────────────────────────────┐
│         Stacked Bidirectional LSTMs (128 -> 64)           │
│         Scans for unnatural temporal speech pauses        │
└─────────────────────────────┬─────────────────────────────┘
                              ▼
┌───────────────────────────────────────────────────────────┐
│       Dense Classification (Sigmoid Output -> Predict)    │
└───────────────────────────────────────────────────────────┘
```

---

## 🔬 Feature Representation
Unlike standard systems that rely purely on Mel spectrograms (which overfit to specific text-to-speech vocoders), VoiceGuard uses a stacked multi-cepstral representation:
- **LFCC (Linear Frequency Cepstral Coefficients)**: Extracted with 60 bands. Linear frequency spacing is exceptional at capturing high-frequency voice cloning artifacts.
- **MFCC (Mel-Frequency Cepstral Coefficients)**: Extracted with 60 bands. Mel spacing models standard human speech timbre.
- **Delta-LFCC**: First-order temporal derivatives of LFCC to capture the dynamic progression of frequency sweeps.

These are stacked to form a single input tensor of shape `(180, 128, 1)`.

---

## 🏗️ CNN + BiLSTM Architecture
1. **CNN Front-End**: 4 convolutional blocks extract high-level frequency patterns and spatial anomalies from the 2D audio representation.
2. **Sequential Reshaping**: Transposes and shapes the flattened spatial-frequency features into a sequence of 15 time steps.
3. **Stacked Bidirectional LSTMs**: Recurrent layers run forward and backward over the sequence. Deepfakes often contain local temporal glitches, unnatural transitions, or phase anomalies. The BiLSTM layers excel at detecting these sequential inconsistencies.

---

## 📁 Project Structure
```text
voiceguard/
├── notebook.ipynb          # Training pipeline notebook
├── app.py                  # Streamlit web application (Premium Cyber Theme)
├── model/
│   ├── best_model.keras    # Trained CNN-BiLSTM Keras model
│   └── model_config.json   # Decision threshold & test set metrics
├── utils/
│   ├── predict.py          # Single file prediction and CLI interface
│   ├── feature_extraction.py # Feature extraction utilities
│   ├── augmentation.py     # Augmentation functions (pitch shift, time stretch)
│   └── metrics.py          # Verification metric calculation utilities
├── requirements.txt        # Project dependencies
└── README.md               # Documentation
```

---

## 🚀 Run Locally

### 1. Set Up Environment
```bash
# Clone the repository
git clone https://github.com/example/voiceguard.git
cd voiceguard

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Inference on a Single File
```bash
python utils/predict.py --audio path/to/sample.wav
```

### 3. Run the Streamlit Dashboard
```bash
streamlit run app.py
```

---

## 🛠️ Tech Stack
| Component | Technology | Description |
| :--- | :--- | :--- |
| **Deep Learning** | TensorFlow / Keras | Model definition and prediction |
| **Audio Processing** | Librosa / SciPy | Feature extraction and DSP calculations |
| **Web Dashboard** | Streamlit | Cyber-theme UI dashboard |
| **Math / Data** | NumPy / Pandas / Scikit-Learn | Data engineering and metrics |
| **Plotting** | Matplotlib | Specshow plots and bar charts |
