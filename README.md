---
tags:
  - text-to-speech
  - qualcomm
  - qnn
  - onnx
  - quantized
license: mit
---

## Note: This repository is a work in progress. The README will be updated with more details on the QCS6490 porting process, model architecture, and calibration data generation workflow in the coming weeks. Please check back for updates!

# Supertonic-2 QNN Inference

<audio controls src="https://huggingface.co/dev-ansh-r/supertonic2-qualcomm-quantized/resolve/main/output/custom.wav"></audio>

High-quality Text-to-Speech synthesis using Supertonic-2 ONNX models.

## Quick Start

```bash
# Basic usage
python supertonic_inference.py --text "Hello world" --voice M1 -o output.wav

# With specific seed for reproducibility
python supertonic_inference.py \
  --text "Your text here" \
  --voice F1 \
  --seed 42 \
  --output output.wav
```

## Features

- ✅ **High Quality**: Matches official Supertonic library behavior
- ✅ **Reproducible**: Use `--seed` for consistent outputs
- ✅ **Multilingual**: Supports 5 languages (EN, KO, ES, PT, FR)
- ✅ **Multiple Voices**: 10 voice styles (5 female, 5 male)
- ✅ **Customizable**: Adjust quality, speed, and more

## QCS6490 Deployment

This repository includes complete resources for porting to **Qualcomm QCS6490** (Hexagon HTP V68):

- **[QCS6490 Porting Guide](QCS6490_PORTING_GUIDE.md)** - Step-by-step deployment guide
- **[Architecture Deep Dive](supertonic_architecture.md)** - Complete model architecture documentation
- **[Calibration Data Guide](CALIBRATION_DATA.md)** - Quantization and validation workflow

### Generate Calibration Data

```bash
# Generate 10 calibration samples for quantization and accuracy validation
python generate_calibration_data.py

# Validate QNN model accuracy against ONNX reference
python validate_qnn_accuracy.py \
  --calibration-dir calibration_data \
  --qnn-dir qnn_outputs \
  --report accuracy_report.json
```

**Expected Performance on QCS6490:**
- 100-150× real-time inference speed
- ~75ms latency for 2-second audio (5 diffusion steps)
- ~70-100 MB total model size (after INT8/FP16 quantization)

## Model Files

Required directory structure:
```
model/
├── onnx/
│   ├── text_encoder.onnx
│   ├── duration_predictor.onnx
│   ├── vector_estimator.onnx
│   ├── vocoder.onnx
│   ├── tts.json
│   └── unicode_indexer.json
└── voice_styles/
    ├── F1.json ... F5.json (Female voices)
    └── M1.json ... M5.json (Male voices)
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--text`, `-t` | Text to synthesize | **Required** |
| `--voice`, `-v` | Voice style (F1-F5, M1-M5) | M1 |
| `--lang`, `-l` | Language (en, ko, es, pt, fr) | en |
| `--output`, `-o` | Output WAV file path | output/output.wav |
| `--steps` | Diffusion steps (more = better) | 10 |
| `--speed` | Speech speed multiplier | 1.0 |
| `--seed` | Random seed (for reproducibility) | None |
| `--quiet`, `-q` | Suppress progress messages | False |

## Examples

### Basic Synthesis
```bash
python supertonic_inference.py \
  --text "The weather is nice today." \
  --voice F1 \
  --output weather.wav
```

### High Quality with Reproducible Output
```bash
python supertonic_inference.py \
  --text "Important announcement." \
  --voice M1 \
  --steps 20 \
  --seed 42 \
  --output announcement.wav
```

### Faster Speech
```bash
python supertonic_inference.py \
  --text "Quick update message." \
  --voice F2 \
  --speed 1.2 \
  --output quick.wav
```

### Spanish Language
```bash
python supertonic_inference.py \
  --text "Hola mundo" \
  --lang es \
  --voice M3 \
  --output spanish.wav
```

## Voice Styles

| Code | Type | Description |
|------|------|-------------|
| F1-F5 | Female | 5 distinct female voices |
| M1-M5 | Male | 5 distinct male voices |

## Supported Languages

- `en` - English
- `ko` - Korean
- `es` - Spanish
- `pt` - Portuguese
- `fr` - French

## Python API

```python
from supertonic_inference import SupertonicTTS, save_wav

# Initialize
tts = SupertonicTTS(model_dir="model/onnx")

# Synthesize
waveform, duration = tts.synthesize(
    text="Hello world",
    voice_name="M1",
    lang="en",
    diffusion_steps=10,
    speed=1.0,
    seed=42
)

# Save
save_wav("output.wav", waveform, tts.sample_rate)
```

## Performance

- **Speed**: ~25× faster than real-time on CPU
- **Quality**: Matches official Supertonic library
- **Sample Rate**: 44.1 kHz
- **Variance**: ~5% duration variance due to diffusion randomness

## Parameters Guide

### Diffusion Steps
- **5-10 steps**: Fast, good quality (default: 10)
- **15-20 steps**: Higher quality, slower
- **Trade-off**: Each step improves quality but increases compute time

### Speed
- **0.8-1.0**: Slower, more natural
- **1.0**: Normal speed (default)
- **1.1-1.5**: Faster, still intelligible

### Seed
- **None**: Different output each run (random)
- **Integer**: Reproducible output with same seed
- **Use case**: Testing, demos, consistent results

## Project Structure

```
supertonic2-qualcomm/
├── supertonic_inference.py    # Main inference script
├── README.md                   # This file
├── model/
│   ├── onnx/                   # ONNX models
│   └── voice_styles/           # Voice embeddings
├── inputs/                     # Test inputs
└── output/                     # Generated audio
```

## Notes

- Text is automatically preprocessed (normalization, punctuation)
- Periods are added automatically if missing
- Unicode characters are handled transparently
- Emoji are automatically removed

## License

This implementation uses models from [Supertone/supertonic-2](https://huggingface.co/Supertone/supertonic-2).

## Reference

Official repository: https://github.com/supertone-inc/supertonic
