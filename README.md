# TinyML Wave Detection for Radar Sensor

Machine learning model for detecting hand waving gestures using BGT60TR13C 60GHz FMCW radar sensor on ARM Cortex-M7 MCU.

## Overview

This project implements a TinyML-based wave detection system that analyzes radar energy patterns to classify:
- **No Presence**: Empty room / no target
- **Waving**: Hand waving gesture detected

## Hardware Target

- **MCU**: ATSAMS70Q21 (ARM Cortex-M7 @ 300MHz)
- **Flash**: 2 MB
- **RAM**: 384 KB
- **Radar**: Infineon BGT60TR13C (60 GHz FMCW)

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                ATSAMS70Q21 (Cortex-M7 @ 300MHz)             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────┐    ┌─────────────┐    ┌─────────────────┐   │
│  │BGT60TR13C│    │   Energy    │    │  TinyML Model   │   │
│  │ (60 GHz) │───>│  Extraction │───>│  Wave Detector  │   │
│  │  Radar   │SPI │             │    │  (Int8, 2.9KB)  │   │
│  └──────────┘    └─────────────┘    └────────┬────────┘   │
│                                               │            │
│                                               ▼            │
│                                     ┌─────────────────┐   │
│                                     │  Wave Detected  │   │
│                                     │  + LED Control  │   │
│                                     └─────────────────┘   │
│                                                            │
└─────────────────────────────────────────────────────────────┘
```

## Model Architecture

Simple dense neural network optimized for MCU deployment:

```
Input (16 energy values)
    ↓
Dense(8, ReLU) ─── 136 params
    ↓
Dense(4, ReLU) ─── 36 params
    ↓
Dense(2, Softmax) ─ 10 params
    ↓
Output (no_presence / waving)

Total: 182 parameters
```

## Model Statistics

| Metric | Value |
|--------|-------|
| Parameters | 182 |
| Model Size (int8) | 2.9 KB |
| Flash Usage | 0.14% of 2 MB |
| RAM (Tensor Arena) | 4 KB |
| Inference Time | <1 ms |
| Training Accuracy | 100% |

## Training Data

Model trained on raw sensor data captured from BGT60TR13C:
- **No presence**: Stable low energy (~270-310)
- **Waving**: Fluctuating high energy (~450-2700)

The key indicator for waving is the variance in energy values - waving produces highly fluctuating readings while no presence shows stable low values.

Window size: 16 consecutive frames analyzed per inference.

## Project Structure

```
tinyml_radar/
├── data_loader.py         # Load CSV sensor data
├── train_model.py         # Model training
├── export_model.py        # TFLite export and quantization
├── requirements.txt       # Python dependencies
├── models/                # Trained models
│   ├── radar_wave_model.keras
│   ├── radar_model_float.tflite
│   └── radar_model_int8.tflite
├── firmware/              # MCU integration code
│   ├── radar_model_data.h # Model as C array
│   ├── tflite_inference.h # Inference API
│   └── tflite_inference.c # Inference implementation
└── *.csv                  # Training data
```

## Usage

### Training

```bash
# Install dependencies
pip install -r requirements.txt

# Train model
python train_model.py

# Export to TFLite
python export_model.py
```

### Firmware Integration

Include the generated files in your MCU project:

```c
#include "radar_model_data.h"
#include "tflite_inference.h"

// Initialize
ml_init();

// Run inference on energy window
float energy_window[16];  // Normalized 0-1
ml_result_t result;
ml_inference(energy_window, &result);

if (result.predicted_class == ML_CLASS_WAVING) {
    // Wave gesture detected
    led_on();
}
```

## Resource Analysis

### Flash Memory

| Component | Size |
|-----------|------|
| TinyML Model | 2.9 KB |
| Firmware Code | ~20 KB |
| **Total** | ~23 KB |
| **Available** | 2,048 KB |
| **Usage** | 1.1% |

### RAM

| Component | Size |
|-----------|------|
| Tensor Arena | 4 KB |
| Energy Buffer | 64 bytes |
| Stack | 4 KB |
| **Total** | ~8 KB |
| **Available** | 384 KB |
| **Usage** | 2.1% |

## Data Format

Input CSV format from sensor:
```
frame_idx,rx_idx,presence_energy
0,0,283.94
1,0,300.73
2,0,312.16
...
```

Energy values are normalized to 0-1 range before inference using min-max scaling.

## Dependencies

- TensorFlow >= 2.10.0
- NumPy >= 1.21.0
- Pandas >= 1.3.0

## Firmware Requirements

- TensorFlow Lite Micro library
- ARM CMSIS headers
