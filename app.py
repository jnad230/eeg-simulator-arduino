import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="EEG Signal Analyzer", layout="wide")

st.title("🧠 EEG Signal Analysis: Normal vs Seizure-Like Activity")
st.markdown("""
Arduino-generated EEG data demonstrating normal brain activity vs seizure-like patterns.
""")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('data/sample_data.csv')
    return df

try:
    df = load_data()
    st.success(f"✅ Loaded {len(df)} samples ({df['timestamp_ms'].max()/1000:.1f} seconds at 100 Hz)")
except:
    st.error("❌ Could not load data")
    st.stop()

# Create figure with 3 subplots
fig = plt.figure(figsize=(16, 10))

# ===== PLOT 1: Full Time Domain Signal =====
ax1 = plt.subplot(3, 1, 1)

# Find transitions
transition_indices = df[df['mode'] != df['mode'].shift()].index.tolist()

# Plot each contiguous segment separately
for i in range(len(transition_indices)):
    start_idx = transition_indices[i]
    end_idx = transition_indices[i+1] if i+1 < len(transition_indices) else len(df)
    
    segment = df.iloc[start_idx:end_idx]
    time_seg = segment['timestamp_ms'] / 1000
    eeg_seg = segment['eeg_value']
    mode = segment.iloc[0]['mode']
    
    color = 'b' if mode == 'normal' else 'r'
    label = 'Normal' if mode == 'normal' and i == 0 else ('Abnormal' if mode == 'abnormal' and color == 'r' else None)
    
    ax1.plot(time_seg, eeg_seg, color=color, linewidth=0.8, alpha=0.7, label=label)

# Mark mode switches
for idx in transition_indices[1:]:
    ax1.axvline(x=df.iloc[idx]['timestamp_ms']/1000, color='orange', linestyle='--', linewidth=2, alpha=0.7)

ax1.set_title('EEG Signal - Time Domain', fontsize=14, fontweight='bold')
ax1.set_xlabel('Time (seconds)', fontsize=12)
ax1.set_ylabel('Amplitude (μV)', fontsize=12)
ax1.set_ylim(-3, 3)
ax1.grid(True, alpha=0.3)

# Add legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='b', lw=2, label='Normal'),
    Line2D([0], [0], color='r', lw=2, label='Abnormal'),
    Line2D([0], [0], color='orange', lw=2, linestyle='--', label='Mode Switch')
]
ax1.legend(handles=legend_elements, loc='upper right')

# ===== PLOT 2: Normal FFT =====
ax2 = plt.subplot(3, 1, 2)

first_normal_block = df[(df['mode'] == 'normal') & (df['timestamp_ms'] < 5000)]
normal_data = first_normal_block['eeg_value'].values

sample_rate = 100
n = len(normal_data)

freqs = np.fft.fftfreq(n, 1/sample_rate)
fft_vals = np.fft.fft(normal_data)
magnitude = 2.0 * np.abs(fft_vals) / n

# Filter to positive frequencies AND 0-40 Hz range
freq_mask = (freqs > 0) & (freqs <= 40)
freq_plot = freqs[freq_mask]
mag_plot = magnitude[freq_mask]

ax2.plot(freq_plot, mag_plot, 'b-', linewidth=2)
ax2.set_title('Frequency Spectrum - NORMAL Activity', fontsize=14, fontweight='bold', color='blue')
ax2.set_xlabel('Frequency (Hz)', fontsize=12)
ax2.set_ylabel('Power', fontsize=12)
ax2.set_xlim(0, 40)
ax2.grid(True, alpha=0.3)

ax2.axvspan(0.5, 4, alpha=0.15, color='purple', label='Delta (0.5-4 Hz)')
ax2.axvspan(4, 8, alpha=0.15, color='blue', label='Theta (4-8 Hz)')
ax2.axvspan(8, 13, alpha=0.15, color='green', label='Alpha (8-13 Hz)')
ax2.axvspan(13, 30, alpha=0.15, color='orange', label='Beta (13-30 Hz)')
ax2.legend(loc='upper right', fontsize=9)

# ===== PLOT 3: Abnormal FFT =====
ax3 = plt.subplot(3, 1, 3)

first_abnormal_block = df[(df['mode'] == 'abnormal') & (df['timestamp_ms'] >= 5000) & (df['timestamp_ms'] < 10000)]
abnormal_data = first_abnormal_block['eeg_value'].values

n_ab = len(abnormal_data)

freqs_ab = np.fft.fftfreq(n_ab, 1/sample_rate)
fft_vals_ab = np.fft.fft(abnormal_data)
magnitude_ab = 2.0 * np.abs(fft_vals_ab) / n_ab

# Filter to positive frequencies AND 0-40 Hz range
freq_mask_ab = (freqs_ab > 0) & (freqs_ab <= 40)
freq_plot_ab = freqs_ab[freq_mask_ab]
mag_plot_ab = magnitude_ab[freq_mask_ab]

ax3.plot(freq_plot_ab, mag_plot_ab, 'r-', linewidth=2)
ax3.set_title('Frequency Spectrum - ABNORMAL Activity (Seizure-Like)', fontsize=14, fontweight='bold', color='red')
ax3.set_xlabel('Frequency (Hz)', fontsize=12)
ax3.set_ylabel('Power', fontsize=12)
ax3.set_xlim(0, 40)
ax3.grid(True, alpha=0.3)

ax3.axvspan(0.5, 4, alpha=0.15, color='purple', label='Delta (0.5-4 Hz)')
ax3.axvspan(4, 8, alpha=0.15, color='blue', label='Theta (4-8 Hz)')
ax3.axvspan(8, 13, alpha=0.15, color='green', label='Alpha (8-13 Hz)')
ax3.axvspan(13, 30, alpha=0.15, color='orange', label='Beta (13-30 Hz)')
ax3.legend(loc='upper right', fontsize=9)

# Add note
fig.text(0.99, 0.01, 
         'Note: Abnormal pattern shows increased low-frequency (Delta) and high-frequency (Beta) power.\n' +
         'This is characteristic of seizure activity in real EEG recordings.',
         ha='right', fontsize=9, style='italic', color='gray')

plt.tight_layout()
st.pyplot(fig)

# ===== INFO SECTION =====
st.markdown("---")
st.markdown("### 📊 Key Observations")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **Normal Activity (Blue):**
    - Low amplitude (~±1 μV)
    - Smooth, rhythmic oscillations
    - Dominant alpha peak at ~10 Hz
    - Minimal low-frequency content
    """)

with col2:
    st.markdown("""
    **Abnormal Activity (Red):**
    - High amplitude (~±3 μV) - 3x increase
    - Chaotic, irregular spikes
    - Strong delta power at ~2 Hz
    - Elevated beta activity at 20-30 Hz
    """)

st.markdown("---")
st.markdown("### 🔬 Technical Details")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **Signal Generation:**
    - Arduino Uno (Wokwi simulation)
    - 100 Hz sampling rate (clinical standard)
    - Mode switches every 5 seconds
    - Realistic noise artifacts
    """)

with col2:
    st.markdown("""
    **Analysis Methods:**
    - Fast Fourier Transform (FFT)
    - Frequency band decomposition
    - Signal quality monitoring
    - Time-domain visualization
    """)

st.markdown("---")
st.markdown("""
⚠️ **Educational demonstration using simulated data - not a medical device**

[View Full Project on GitHub](https://github.com/jnad230/eeg-simulator-arduino)
""")