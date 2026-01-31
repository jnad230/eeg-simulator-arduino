"""
EEG Signal Viewer with Frequency Analysis (FFT)
Shows time-domain signal AND frequency spectrum
"""

import matplotlib.pyplot as plt
import numpy as np
import csv
from scipy import signal as sp_signal

# Configuration
CSV_FILE = '../data/sample_data.csv'

# Read data
timestamps = []
eeg_values = []
quality_scores = []
modes = []

with open(CSV_FILE, 'r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip header
    
    for row in reader:
        timestamps.append(float(row[0]) / 1000.0)
        eeg_values.append(float(row[1]))
        quality_scores.append(int(row[3]))
        modes.append(row[4])

# Convert to numpy arrays
timestamps = np.array(timestamps)
eeg_values = np.array(eeg_values)
sample_rate = 100  # Hz

# Find transition point between normal and abnormal
transition_idx = 0
for i, mode in enumerate(modes):
    if mode == 'abnormal':
        transition_idx = i
        break

# Split data
normal_eeg = eeg_values[:transition_idx]
abnormal_eeg = eeg_values[transition_idx:transition_idx+1000]  # Take 10 seconds

# Compute FFT for both segments
def compute_fft(data, sample_rate):
    """Compute frequency spectrum using FFT"""
    n = len(data)
    fft_vals = np.fft.fft(data)
    fft_freq = np.fft.fftfreq(n, 1/sample_rate)
    
    # Only positive frequencies
    positive_freq_idx = fft_freq > 0
    frequencies = fft_freq[positive_freq_idx]
    magnitude = np.abs(fft_vals[positive_freq_idx]) / n
    
    return frequencies, magnitude

normal_freq, normal_mag = compute_fft(normal_eeg, sample_rate)
abnormal_freq, abnormal_mag = compute_fft(abnormal_eeg, sample_rate)

# Create figure with 3 subplots
fig = plt.figure(figsize=(14, 10))

# Plot 1: Time-domain signal (top)
ax1 = plt.subplot(3, 1, 1)
ax1.plot(timestamps[:transition_idx], eeg_values[:transition_idx], 
         'b-', linewidth=0.8, label='Normal', alpha=0.7)
ax1.plot(timestamps[transition_idx:], eeg_values[transition_idx:], 
         'r-', linewidth=0.8, label='Abnormal', alpha=0.7)
ax1.axvline(x=timestamps[transition_idx], color='orange', 
            linestyle='--', linewidth=2, label='Mode Switch')
ax1.set_title('EEG Signal - Time Domain', fontsize=14, fontweight='bold')
ax1.set_ylabel('Amplitude (μV)')
ax1.set_xlabel('Time (seconds)')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# Plot 2: Frequency spectrum - Normal (middle)
ax2 = plt.subplot(3, 1, 2)
ax2.plot(normal_freq, normal_mag, 'b-', linewidth=1.5)
ax2.set_title('Frequency Spectrum - NORMAL Activity', fontsize=14, fontweight='bold')
ax2.set_ylabel('Power')
ax2.set_xlabel('Frequency (Hz)')
ax2.set_xlim(0, 40)
ax2.grid(True, alpha=0.3)

# Add frequency band annotations
ax2.axvspan(0.5, 4, alpha=0.2, color='purple', label='Delta (0.5-4 Hz)')
ax2.axvspan(4, 8, alpha=0.2, color='blue', label='Theta (4-8 Hz)')
ax2.axvspan(8, 13, alpha=0.2, color='green', label='Alpha (8-13 Hz)')
ax2.axvspan(13, 30, alpha=0.2, color='orange', label='Beta (13-30 Hz)')
ax2.legend(loc='upper right', fontsize=8)

# Plot 3: Frequency spectrum - Abnormal (bottom)
ax3 = plt.subplot(3, 1, 3)
ax3.plot(abnormal_freq, abnormal_mag, 'r-', linewidth=1.5)
ax3.set_title('Frequency Spectrum - ABNORMAL Activity (Seizure-Like)', 
              fontsize=14, fontweight='bold', color='red')
ax3.set_ylabel('Power')
ax3.set_xlabel('Frequency (Hz)')
ax3.set_xlim(0, 40)
ax3.grid(True, alpha=0.3)

# Add frequency band annotations
ax3.axvspan(0.5, 4, alpha=0.2, color='purple', label='Delta (0.5-4 Hz)')
ax3.axvspan(4, 8, alpha=0.2, color='blue', label='Theta (4-8 Hz)')
ax3.axvspan(8, 13, alpha=0.2, color='green', label='Alpha (8-13 Hz)')
ax3.axvspan(13, 30, alpha=0.2, color='orange', label='Beta (13-30 Hz)')
ax3.legend(loc='upper right', fontsize=8)

# Add text annotations
fig.text(0.99, 0.02, 
         'Note: Abnormal pattern shows increased low-frequency (Delta) and high-frequency (Beta) power\n' +
         'This is characteristic of seizure activity in real EEG recordings',
         ha='right', fontsize=9, style='italic', color='gray')

plt.tight_layout()
plt.show()

print("✅ FFT Analysis Complete!")
print(f"📊 Normal segment: {len(normal_eeg)} samples")
print(f"📊 Abnormal segment: {len(abnormal_eeg)} samples")
print(f"🔬 Frequency resolution: {sample_rate/len(normal_eeg):.2f} Hz")