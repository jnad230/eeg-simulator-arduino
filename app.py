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

time = df['timestamp_ms'] / 1000
normal_mask = df['mode'] == 'normal'

ax1.plot(time[normal_mask], df['eeg_value'][normal_mask], 'b-', linewidth=0.8, alpha=0.7, label='Normal')
ax1.plot(time[~normal_mask], df['eeg_value'][~normal_mask], 'r-', linewidth=0.8, alpha=0.7, label='Abnormal')

# Mark mode switches
transitions = df[df['mode'] != df['mode'].shift()].index
for idx in transitions[1:]:  # Skip first
    ax1.axvline(x=df.iloc[idx]['timestamp_ms']/1000, color='orange', linestyle='--', linewidth=2, alpha=0.7)

ax1.set_title('EEG Signal - Time Domain', fontsize=14, fontweight='bold')
ax1.set_xlabel('Time (seconds)', fontsize=12)
ax1.set_ylabel('Amplitude (μV)', fontsize=12)
ax1.set_ylim(-3, 3)
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# Add legend for mode switch
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='b', lw=2, label='Normal'),
    Line2D([0], [0], color='r', lw=2, label='Abnormal'),
    Line2D([0], [0], color='orange', lw=2, linestyle='--', label='Mode Switch')
]
ax1.legend(handles=legend_elements, loc='upper right')

# ===== PLOT 2: Normal FFT =====
ax2 = plt.subplot(3, 1, 2)

normal_data = df[df['mode'] == 'normal']['eeg_value'].values[:1000]
sample_rate = 100

freqs = np.fft.fftfreq(len(normal_data), 1/sample_rate)
fft_vals = np.fft.fft(normal_data)
magnitude = np.abs(fft_vals) / len(normal_data)

positive_mask = freqs > 0
freq_plot = freqs[positive_mask][:200]
mag_plot = magnitude[positive_mask][:200]

ax2.plot(freq_plot, mag_plot, 'b-', linewidth=2)
ax2.set_title('Frequency Spectrum - NORMAL Activity', fontsize=14, fontweight='bold', color='blue')
ax2.set_xlabel('Frequency (Hz)', fontsize=12)
ax2.set_ylabel('Power', fontsize=12)
ax2.set_xlim(0, 40)
ax2.grid(True, alpha=0.3)

# Add frequency bands
ax2.axvspan(0.5, 4, alpha=0.15, color='purple', label='Delta (0.5-4 Hz)')
ax2.axvspan(4, 8, alpha=0.15, color='blue', label='Theta (4-8 Hz)')
ax2.axvspan(8, 13, alpha=0.15, color='green', label='Alpha (8-13 Hz)')
ax2.axvspan(13, 30, alpha=0.15, color='orange', label='Beta (13-30 Hz)')
ax2.legend(loc='upper right', fontsize=9)

# ===== PLOT 3: Abnormal FFT =====
ax3 = plt.subplot(3, 1, 3)

abnormal_data = df[df['mode'] == 'abnormal']['eeg_value'].values[:1000]

freqs_ab = np.fft.fftfreq(len(abnormal_data), 1/sample_rate)
fft_vals_ab = np.fft.fft(abnormal_data)
magnitude_ab = np.abs(fft_vals_ab) / len(abnormal_data)

positive_mask_ab = freqs_ab > 0
freq_plot_ab = freqs_ab[positive_mask_ab][:200]
mag_plot_ab = magnitude_ab[positive_mask_ab][:200]

ax3.plot(freq_plot_ab, mag_plot_ab, 'r-', linewidth=2)
ax3.set_title('Frequency Spectrum - ABNORMAL Activity (Seizure-Like)', fontsize=14, fontweight='bold', color='red')
ax3.set_xlabel('Frequency (Hz)', fontsize=12)
ax3.set_ylabel('Power', fontsize=12)
ax3.set_xlim(0, 40)
ax3.grid(True, alpha=0.3)

# Add frequency bands
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