# EEG Simulator with Frequency Analysis

A demonstration of embedded signal processing for medical devices, simulating wearable EEG with real-time quality monitoring and frequency analysis.

---

## What It Does

Simulates a brain activity monitor that:
- Generates realistic EEG-like signals on Arduino
- Tracks signal quality (motion artifacts, noise)
- Analyzes frequency patterns (Normal vs Seizure-like)
- Visualizes results in Python

---

## Why It Matters

**Signal Quality:** Real wearable EEG devices need to detect when data is corrupted by motion or poor electrode contact.

**Frequency Analysis:** Seizure detection relies on identifying abnormal frequency patterns - increased low-frequency (delta) and high-frequency (beta) activity compared to normal alpha-dominated rhythms.

---

## Technical Stack

- **Hardware:** Arduino Uno (Wokwi simulator)
- **Firmware:** C++ signal generation at 100 Hz
- **Analysis:** Python with NumPy, SciPy, Matplotlib
- **Signal Processing:** FFT for frequency domain analysis

---

## Key Results

### Normal Brain Activity
- Dominant alpha rhythm (~10 Hz) - relaxed, awake state
- Low noise, stable signal quality

### Abnormal Activity (Seizure-like)
- High delta power (~2 Hz) - pathological slow waves
- Increased beta (~20 Hz) - fast spiking
- Elevated amplitude and broad-spectrum activity

These patterns match real clinical EEG characteristics documented in neurophysiology literature.


```

**Built with:** Arduino, Python, FFT  
**Inspired by:** Seer Medical's wearable EEG technology

