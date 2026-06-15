import os
import json
import tempfile
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import librosa
import librosa.display
from utils.predict import predict_single, compute_lfcc, SR, DURATION, HOP_LENGTH, N_FFT, N_LFCC, N_MFCC

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VoiceGuard — Cybernetic Deepfake Audio Detector",
    page_icon="🎙️",
    layout="wide"
)

# ── Custom CSS (Cyberpunk Dark Mode / Glassmorphism) ──────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=Space+Mono:wght@400;700&display=swap');
    
    /* Global Styles */
    .stApp {
        background-color: #080B10;
        color: #E6EDF3;
        font-family: 'Outfit', sans-serif;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #0E131F;
        border-right: 1px solid #1E293B;
    }
    
    /* Custom Headers */
    h1, h2, h3 {
        color: #00F2FE !important;
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    
    /* Neon Text Glowing effect */
    .glow-text {
        text-shadow: 0 0 10px rgba(0, 242, 254, 0.5);
    }
    
    /* Glassmorphism Cards */
    .premium-card {
        background: rgba(14, 19, 31, 0.8);
        border: 1px solid rgba(30, 41, 59, 0.8);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        backdrop-filter: blur(4px);
        -webkit-backdrop-filter: blur(4px);
    }
    
    /* Metric Cards */
    .metric-container {
        display: flex;
        justify-content: space-around;
        gap: 15px;
        margin-top: 15px;
        flex-wrap: wrap;
    }
    .metric-card-custom {
        background: rgba(15, 23, 42, 0.65);
        border: 1px solid rgba(0, 242, 254, 0.2);
        border-radius: 10px;
        padding: 18px;
        text-align: center;
        flex: 1;
        min-width: 130px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.25);
        transition: all 0.3s ease;
    }
    .metric-card-custom:hover {
        border: 1px solid rgba(0, 242, 254, 0.6);
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0, 242, 254, 0.15);
    }
    .metric-value-custom {
        font-size: 26px;
        font-weight: 700;
        color: #00F2FE;
        font-family: 'Space Mono', monospace;
    }
    .metric-label-custom {
        font-size: 12px;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 5px;
    }
    
    /* Verdict boxes */
    .verdict-box {
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        font-weight: 700;
        font-size: 24px;
        margin-top: 15px;
        margin-bottom: 25px;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    .verdict-genuine {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(5, 150, 105, 0.2) 100%);
        border: 2px solid #10B981;
        color: #34D399;
        text-shadow: 0 0 10px rgba(52, 211, 153, 0.3);
    }
    .verdict-deepfake {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.1) 0%, rgba(220, 38, 38, 0.2) 100%);
        border: 2px solid #EF4444;
        color: #F87171;
        text-shadow: 0 0 10px rgba(248, 113, 113, 0.3);
    }
    
    /* Custom buttons */
    .stButton>button {
        background: linear-gradient(90deg, #4F46E5 0%, #06B6D4 100%) !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        font-family: 'Outfit', sans-serif !important;
        border: none !important;
        padding: 12px 28px !important;
        border-radius: 8px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(6, 182, 212, 0.3) !important;
    }
    .stButton>button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 8px 25px rgba(6, 182, 212, 0.5) !important;
    }
    .stButton>button:active {
        transform: translateY(1px) !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Model files verification ───────────────────────────────────────────────
MODEL_PATH = "model/best_model.keras"
CONFIG_PATH = "model/model_config.json"
model_ok = os.path.exists(MODEL_PATH) and os.path.exists(CONFIG_PATH)

# ── Sidebar Configuration ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<div style='text-align: center; margin-bottom: 15px;'>", unsafe_allow_html=True)
    st.image("https://img.icons8.com/nolan/128/microphone.png", width=80)
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<h2 style='text-align: center; margin-top: 0;'>VOICEGUARD</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #94A3B8; font-size: 13px; margin-top: -10px;'>Detect the Deception, Trust the Voice</p>", unsafe_allow_html=True)
    st.markdown("<hr style='border-color: #1E293B;'/>", unsafe_allow_html=True)
    
    # Model evaluation metrics
    st.subheader("🤖 Model Benchmarks")
    st.caption("Evaluated on the Fake-or-Real (FoR) test set:")
    
    col_side1, col_side2 = st.columns(2)
    with col_side1:
        st.metric(label="Overall Accuracy", value="89.56%")
        st.metric(label="F1 Score", value="89.12%")
        st.metric(label="Genuine Acc", value="98.24%")
    with col_side2:
        st.metric(label="Equal Error Rate", value="6.24%")
        st.metric(label="ROC AUC", value="96.84%")
        st.metric(label="Deepfake Acc", value="81.28%")
        
    st.markdown("<hr style='border-color: #1E293B;'/>", unsafe_allow_html=True)
    
    st.subheader("🧬 System Parameters")
    st.markdown(f"""
    - **Model type**: CNN + Stacked BiLSTM
    - **Acoustic Features**: 
      * LFCC (60 bands)
      * MFCC (60 bands)
      * Delta-LFCC (60 bands)
    - **Input dimension**: `(180, 128, 1)`
    - **Target Sample Rate**: {SR} Hz
    - **Analysis Window**: {DURATION}s
    """)
    st.markdown("<hr style='border-color: #1E293B;'/>", unsafe_allow_html=True)
    st.markdown("<div style='text-align: center; font-size: 11px; color: #64748B;'>VoiceGuard Engine v1.2</div>", unsafe_allow_html=True)

# ── Main Dashboard ─────────────────────────────────────────────────────────
st.markdown("<h1 class='glow-text'>🎙️ VoiceGuard Deepfake Audio Detection</h1>", unsafe_allow_html=True)
st.write("Upload a voice recording below. Our hybrid spatial-temporal deep neural network will extract cepstral descriptors and scan for text-to-speech signatures, voice morphing, and artificial anomalies.")

# Warning if model not found
if not model_ok:
    st.error(
        f"🚨 Deep learning model or configuration files not found in the `model/` directory! "
        f"Ensure `{MODEL_PATH}` and `{CONFIG_PATH}` are present."
    )
    st.stop()

st.markdown("<div class='premium-card'>", unsafe_allow_html=True)
st.subheader("📤 Upload Audio Clip")
uploaded_file = st.file_uploader(
    "Drag and drop or select file (WAV, MP3, FLAC)", 
    type=["wav", "mp3", "flac"]
)
st.markdown("</div>", unsafe_allow_html=True)

if uploaded_file is not None:
    # Save uploaded file temporarily
    file_suffix = f".{uploaded_file.name.split('.')[-1]}"
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    # Extract audio stats
    try:
        y_raw, sr_raw = librosa.load(tmp_path, sr=None)
        duration_sec = librosa.get_duration(y=y_raw, sr=sr_raw)
        
        # Audio Property Display Grid
        col_prop1, col_prop2 = st.columns([1, 1])
        with col_prop1:
            st.markdown("<div class='premium-card'>", unsafe_allow_html=True)
            st.markdown("### 📋 File Properties")
            st.markdown(f"**Filename**: `{uploaded_file.name}`")
            st.markdown(f"**Size**: `{uploaded_file.size / 1024:.2f} KB`")
            st.markdown(f"**Original Sample Rate**: `{sr_raw} Hz`")
            st.markdown(f"**Duration**: `{duration_sec:.2f} seconds`")
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col_prop2:
            st.markdown("<div class='premium-card' style='height: 100%; display: flex; flex-direction: column; justify-content: center;'>", unsafe_allow_html=True)
            st.markdown("### 🔊 Audio Playback")
            st.write("Listen to the uploaded audio clip:")
            st.audio(uploaded_file)
            st.markdown("</div>", unsafe_allow_html=True)
            
    except Exception as e:
        st.error(f"Failed to load audio file: {e}")
        st.stop()

    # ── Inference Triggers ──────────────────────────────────────────────────
    analyze_btn = st.button("🔍 Scan Audio for Deepfake Anomalies")
    
    if analyze_btn:
        with st.spinner("Extracting LFCC + MFCC + Delta feature tensors and running CNN-BiLSTM classifier..."):
            try:
                # Perform inference
                result = predict_single(tmp_path, model_path=MODEL_PATH, config_path=CONFIG_PATH)
                
                # Display Verdict
                is_fake = "Deepfake" in result['label']
                verdict_class = "verdict-deepfake" if is_fake else "verdict-genuine"
                verdict_text = "🚨 AI-GENERATED DEEPFAKE DETECTED" if is_fake else "✅ GENUINE SPEECH DETECTED"
                
                st.markdown(f"""
                <div class="verdict-box {verdict_class}">
                    {verdict_text}
                </div>
                """, unsafe_allow_html=True)
                
                # Metric Cards Grid
                st.markdown("""
                <div class="metric-container">
                    <div class="metric-card-custom">
                        <div class="metric-value-custom">{}</div>
                        <div class="metric-label-custom">Verdict</div>
                    </div>
                    <div class="metric-card-custom">
                        <div class="metric-value-custom">{:.2f}%</div>
                        <div class="metric-label-custom">Confidence Score</div>
                    </div>
                    <div class="metric-card-custom">
                        <div class="metric-value-custom">{:.4f}</div>
                        <div class="metric-label-custom">Deepfake Prob</div>
                    </div>
                    <div class="metric-card-custom">
                        <div class="metric-value-custom">{:.4f}</div>
                        <div class="metric-label-custom">Genuine Prob</div>
                    </div>
                </div>
                """.format(
                    "Deepfake" if is_fake else "Genuine",
                    result['confidence'],
                    result['fake_probability'],
                    result['real_probability']
                ), unsafe_allow_html=True)
                
                st.markdown("<br/>", unsafe_allow_html=True)
                
                # Probability distribution visualizer
                col_bar1, col_bar2 = st.columns([1, 4])
                with col_bar1:
                    st.write("#### Confidence Scale:")
                with col_bar2:
                    st.progress(result['fake_probability'])
                    st.caption(f"Deepfake: {result['fake_probability']*100:.2f}% | Genuine: {result['real_probability']*100:.2f}% (Decision Threshold: {result['threshold_used']})")
                
                # ── Visualizations and Tabs ─────────────────────────────────────────
                st.markdown("<hr style='border-color: #1E293B;'/>", unsafe_allow_html=True)
                st.subheader("📊 Advanced Feature Verification")
                
                tab_wave, tab_spect, tab_stats, tab_arch = st.tabs([
                    "🎵 Raw Waveform", 
                    "💎 Acoustic Spectrograms", 
                    "🧬 Feature Statistics", 
                    "🧠 Neural Architecture"
                ])
                
                # Load audio resampled to 16kHz for uniform processing
                y_16k, _ = librosa.load(tmp_path, sr=SR, duration=DURATION)
                y_padded = np.pad(y_16k, (0, max(0, int(SR * DURATION) - len(y_16k))))[:int(SR * DURATION)]
                
                with tab_wave:
                    st.markdown("### Audio Waveform (Resampled to 16kHz)")
                    fig_wave, ax_wave = plt.subplots(figsize=(12, 3))
                    times = np.linspace(0, len(y_padded) / SR, len(y_padded))
                    
                    # Style the plot
                    plt.style.use('dark_background')
                    fig_wave.patch.set_facecolor('#080B10')
                    ax_wave.set_facecolor('#0E131F')
                    
                    # Choose color based on prediction
                    line_color = '#EF4444' if is_fake else '#10B981'
                    ax_wave.plot(times, y_padded, linewidth=0.6, color=line_color)
                    ax_wave.set_xlabel('Time (seconds)', color='#94A3B8', fontsize=10)
                    ax_wave.set_ylabel('Amplitude', color='#94A3B8', fontsize=10)
                    ax_wave.set_title('Acoustic Waveform', color='#FFFFFF', fontsize=12)
                    ax_wave.grid(True, alpha=0.15, color='#334155')
                    ax_wave.tick_params(colors='#94A3B8')
                    plt.tight_layout()
                    st.pyplot(fig_wave)
                    plt.close()
                    
                with tab_spect:
                    st.markdown("### Acoustic Feature Maps")
                    st.write("These features represent the spectral and timbral fingerprint extracted from the audio and stacked for the neural network input.")
                    
                    # Extract LFCC and MFCC using our utilities
                    lfcc = compute_lfcc(y_padded)
                    mfcc = librosa.feature.mfcc(y=y_padded, sr=SR, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
                    
                    col_sp1, col_sp2 = st.columns(2)
                    with col_sp1:
                        st.write("#### Linear Frequency Cepstral Coefficients (LFCC)")
                        fig_lf, ax_lf = plt.subplots(figsize=(6, 4))
                        fig_lf.patch.set_facecolor('#080B10')
                        ax_lf.set_facecolor('#0E131F')
                        
                        img_lf = librosa.display.specshow(lfcc, sr=SR, hop_length=HOP_LENGTH, x_axis='time', ax=ax_lf, cmap='magma')
                        ax_lf.set_title('LFCC Spectrogram (60 bands)', color='#FFFFFF')
                        ax_lf.tick_params(colors='#94A3B8')
                        fig_lf.colorbar(img_lf, ax=ax_lf, format='%+2.0f dB')
                        plt.tight_layout()
                        st.pyplot(fig_lf)
                        plt.close()
                        
                    with col_sp2:
                        st.write("#### Mel Frequency Cepstral Coefficients (MFCC)")
                        fig_mf, ax_mf = plt.subplots(figsize=(6, 4))
                        fig_mf.patch.set_facecolor('#080B10')
                        ax_mf.set_facecolor('#0E131F')
                        
                        img_mf = librosa.display.specshow(mfcc, sr=SR, hop_length=HOP_LENGTH, x_axis='time', ax=ax_mf, cmap='viridis')
                        ax_mf.set_title('MFCC Spectrogram (60 bands)', color='#FFFFFF')
                        ax_mf.tick_params(colors='#94A3B8')
                        fig_mf.colorbar(img_mf, ax=ax_mf)
                        plt.tight_layout()
                        st.pyplot(fig_mf)
                        plt.close()
                        
                with tab_stats:
                    st.markdown("### Cepstral Distribution Statistics")
                    col_st1, col_st2 = st.columns(2)
                    with col_st1:
                        st.write("#### Average LFCC Coefficients")
                        mean_lfcc = np.mean(lfcc, axis=1)
                        fig_lbar, ax_lbar = plt.subplots(figsize=(6, 4))
                        fig_lbar.patch.set_facecolor('#080B10')
                        ax_lbar.set_facecolor('#0E131F')
                        
                        ax_lbar.bar(range(len(mean_lfcc)), mean_lfcc, color='#00F2FE')
                        ax_lbar.set_title('Average LFCC Band Energy', color='#FFFFFF')
                        ax_lbar.set_xlabel('Coefficient Index', color='#94A3B8')
                        ax_lbar.set_ylabel('Mean Power', color='#94A3B8')
                        ax_lbar.tick_params(colors='#94A3B8')
                        ax_lbar.grid(True, alpha=0.1, color='#334155')
                        plt.tight_layout()
                        st.pyplot(fig_lbar)
                        plt.close()
                        
                    with col_st2:
                        st.write("#### Average MFCC Coefficients")
                        mean_mfcc = np.mean(mfcc, axis=1)
                        fig_mbar, ax_mbar = plt.subplots(figsize=(6, 4))
                        fig_mbar.patch.set_facecolor('#080B10')
                        ax_mbar.set_facecolor('#0E131F')
                        
                        ax_mbar.bar(range(len(mean_mfcc)), mean_mfcc, color='#8A2BE2')
                        ax_mbar.set_title('Average MFCC Band Energy', color='#FFFFFF')
                        ax_mbar.set_xlabel('Coefficient Index', color='#94A3B8')
                        ax_mbar.set_ylabel('Mean Power', color='#94A3B8')
                        ax_mbar.tick_params(colors='#94A3B8')
                        ax_mbar.grid(True, alpha=0.1, color='#334155')
                        plt.tight_layout()
                        st.pyplot(fig_mbar)
                        plt.close()
                        
                    st.write("#### Audio Waveform Parameters")
                    stats_data = pd.DataFrame({
                        'Parameter': ['Peak Amplitude', 'RMS Energy (Mean)', 'Zero Crossing Rate (Mean)', 'Silence Ratio (Frames < 0.01)'],
                        'Value': [
                            float(np.max(np.abs(y_padded))),
                            float(np.mean(librosa.feature.rms(y=y_padded))),
                            float(np.mean(librosa.feature.zero_crossing_rate(y=y_padded))),
                            float(np.sum(np.abs(y_padded) < 0.01) / len(y_padded))
                        ]
                    })
                    st.dataframe(stats_data, use_container_width=True)
                    
                with tab_arch:
                    st.markdown("### Neural Network Configuration & Architecture")
                    st.write("The detection engine feeds stacked feature vectors into a deep convolutional network to resolve spectral anomalies, followed by sequential recurrent processing to verify temporal consistency.")
                    
                    st.markdown(r"""
                    #### 1. Input Stacking & Preprocessing
                    Raw audio is loaded at `16,000 Hz` and padded to `4.0 seconds`. It runs through a pre-emphasis filter ($y[t] = y[t] - 0.97 \cdot y[t-1]$) to boost high-frequency regions where AI-generated voice models often leave artifacts.
                    The system extracts:
                    - **LFCC (60 coefficients)**: Linear spacing helps capture high-frequency patterns and speaker spoofing artifacts.
                    - **MFCC (60 coefficients)**: Mel-scale spacing helps capture standard timbral vocal profiles.
                    - **Delta-LFCC (60 coefficients)**: Captures dynamic speed and acceleration of vocal changes.
                    
                    These features are combined into a single matrix of shape `(180, 128, 1)`.
                    
                    #### 2. Model Structure Flowchart
                    """)
                    
                    st.code("""
Input Feature Tensor: (180, 128, 1)
  │
  ▼
Conv2D (32, 3x3) ──> BN ──> Swish ──> MaxPooling2D (2,2) ──> SpatialDropout2D (0.1)
  │
  ▼
Conv2D (64, 3x3) ──> BN ──> Swish ──> MaxPooling2D (2,2) ──> SpatialDropout2D (0.1)
  │
  ▼
Conv2D (128, 3x3) ──> BN ──> Swish ──> MaxPooling2D (3,2) ──> SpatialDropout2D (0.1)
  │
  ▼
Conv2D (128, 3x3) ──> BN ──> Swish ──> MaxPooling2D (1,2) ──> SpatialDropout2D (0.1)
  │
  ▼
Reshape Output Tensor to Sequential Form: (15, 1024)
  │
  ▼
Bidirectional LSTM (128 units, recurrent dropout = 0.1)
  │
  ▼
Bidirectional LSTM (64 units, dropout = 0.2)
  │
  ▼
Dense Classification Block (128 units) ──> Dropout (0.4) ──> Dense Output Layer (1 unit, Sigmoid)
                    """, language="text")
                    
            except Exception as e:
                st.error(f"An error occurred during verification: {e}")
            finally:
                # Clean up temporary file
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

# ── Footer ──────────────────────────────────────────────────────────────────
st.markdown("<hr style='border-color: #1E293B;'/>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #64748B; font-size: 12px;'>🎙️ VoiceGuard Deepfake Audio Detection Engine · MARS Open Projects 2026</p>", unsafe_allow_html=True)
