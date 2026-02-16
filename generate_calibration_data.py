#!/usr/bin/env python3
"""
Generate Calibration Data for Supertonic-2 ONNX Models

Creates representative input/output pairs for:
1. duration_predictor.onnx
2. text_encoder.onnx
3. vector_estimator.onnx (multiple denoising steps)
4. vocoder.onnx

Used for:
- Quantization calibration (QNN/SNPE conversion)
- Accuracy validation after porting to QCS6490
- Performance profiling

Output structure:
calibration_data/
├── sample_001/
│   ├── metadata.json
│   ├── duration_predictor/
│   │   ├── input_text_ids.raw
│   │   ├── input_style_dp.raw
│   │   ├── input_text_mask.raw
│   │   └── output_duration.raw
│   ├── text_encoder/
│   │   ├── input_text_ids.raw
│   │   ├── input_style_ttl.raw
│   │   ├── input_text_mask.raw
│   │   └── output_text_emb.raw
│   ├── vector_estimator/
│   │   ├── step_000/
│   │   │   ├── input_noisy_latent.raw
│   │   │   ├── input_text_emb.raw
│   │   │   ├── input_style_ttl.raw
│   │   │   ├── input_text_mask.raw
│   │   │   ├── input_latent_mask.raw
│   │   │   ├── input_current_step.raw
│   │   │   ├── input_total_step.raw
│   │   │   └── output_latent.raw
│   │   ├── step_001/...
│   │   └── step_N/...
│   └── vocoder/
│       ├── input_latent.raw
│       └── output_waveform.raw
└── sample_002/...
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from supertonic_inference import SupertonicTTS


# Calibration test cases with varying complexity
# All English, F1 voice, 10 diffusion steps
CALIBRATION_SAMPLES = [
    # Short utterances (5-10 words)
    {
        "text": "Hello world, this is a test.",
        "voice": "F1",
        "lang": "en",
        "steps": 10,
        "seed": 42,
        "category": "short"
    },
    {
        "text": "Good morning everyone.",
        "voice": "F1",
        "lang": "en",
        "steps": 10,
        "seed": 100,
        "category": "short"
    },
    {
        "text": "The weather is nice today.",
        "voice": "F1",
        "lang": "en",
        "steps": 10,
        "seed": 150,
        "category": "short"
    },

    # Medium utterances (15-25 words)
    {
        "text": "The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet.",
        "voice": "F1",
        "lang": "en",
        "steps": 10,
        "seed": 200,
        "category": "medium"
    },
    {
        "text": "Machine learning models require careful calibration to ensure accuracy on edge devices.",
        "voice": "F1",
        "lang": "en",
        "steps": 10,
        "seed": 300,
        "category": "medium"
    },
    {
        "text": "Text to speech synthesis has improved dramatically with the advent of neural networks and deep learning.",
        "voice": "F1",
        "lang": "en",
        "steps": 10,
        "seed": 350,
        "category": "medium"
    },

    # Longer utterances (35-50 words)
    {
        "text": "Artificial intelligence has revolutionized many aspects of our daily lives, from smartphone assistants to recommendation systems. The deployment of neural networks on edge devices requires careful optimization, including quantization and calibration, to maintain accuracy while meeting strict latency and power constraints.",
        "voice": "F1",
        "lang": "en",
        "steps": 10,
        "seed": 400,
        "category": "long"
    },
    {
        "text": "Qualcomm's Hexagon processor provides dedicated hardware acceleration for neural network inference. By leveraging the tensor processing units and optimized kernels, developers can achieve real-time performance for complex models like text-to-speech systems on mobile and edge devices.",
        "voice": "F1",
        "lang": "en",
        "steps": 10,
        "seed": 450,
        "category": "long"
    },

    # Edge cases and special content
    {
        "text": "Numbers: 1, 2, 3, 4, 5. Dates: January 1st, 2024. Time: 3:45 PM.",
        "voice": "F1",
        "lang": "en",
        "steps": 10,
        "seed": 500,
        "category": "numbers"
    },
    {
        "text": "Dr. Smith's presentation at 9:00 AM covered topics including: AI, ML, NLP, and TTS. The conference runs from Mon. to Fri.",
        "voice": "F1",
        "lang": "en",
        "steps": 10,
        "seed": 600,
        "category": "abbreviations"
    },
]


class CalibrationDataGenerator:
    """Generate and save calibration data for all 4 ONNX models"""

    def __init__(self, model_dir: str = "model/onnx", output_dir: str = "calibration_data"):
        self.tts = SupertonicTTS(model_dir=model_dir)
        self.output_dir = Path(output_dir)

    def save_tensor(self, filepath: Path, tensor: np.ndarray):
        """Save tensor as raw binary file with shape metadata"""
        # Save raw data - preserve original dtype
        tensor.tofile(filepath)

        # Save shape metadata as JSON
        shape_file = filepath.with_suffix('.shape.json')
        metadata = {
            'shape': list(tensor.shape),
            'dtype': str(tensor.dtype),
            'size_bytes': tensor.nbytes
        }
        with open(shape_file, 'w') as f:
            json.dump(metadata, f, indent=2)

    def generate_sample(self, sample_config: Dict, sample_idx: int):
        """Generate calibration data for one sample"""

        sample_dir = self.output_dir / f"sample_{sample_idx:03d}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        # Save sample metadata
        metadata_file = sample_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(sample_config, f, indent=2)

        print(f"\n{'='*80}")
        print(f"Sample {sample_idx:03d}: {sample_config['category']}")
        print(f"Text: '{sample_config['text'][:60]}...'")
        print(f"Voice: {sample_config['voice']} | Lang: {sample_config['lang']} | Steps: {sample_config['steps']}")
        print(f"{'='*80}")

        # Load voice style
        style_ttl, style_dp = self.tts.load_voice_style(sample_config['voice'])

        # Step 1: Text to IDs (preprocessing)
        print("[1/4] Text preprocessing...")
        text_ids, text_mask = self.tts.text_to_ids(sample_config['text'], sample_config['lang'])
        original_len = text_ids.shape[1]
        print(f"  Tokens: {original_len}")

        # Pad/truncate to fixed length 128 for QNN conversion
        max_len = 128
        if original_len > max_len:
            print(f"  WARNING: Truncating from {original_len} to {max_len}")
            text_ids = text_ids[:, :max_len]
            text_mask = text_mask[:, :, :max_len]
        elif original_len < max_len:
            # Pad text_ids with zeros
            padding = np.zeros((1, max_len - original_len), dtype=text_ids.dtype)
            text_ids = np.concatenate([text_ids, padding], axis=1)
            # Pad text_mask with zeros
            padding_mask = np.zeros((1, 1, max_len - original_len), dtype=text_mask.dtype)
            text_mask = np.concatenate([text_mask, padding_mask], axis=2)
            print(f"  Padded from {original_len} to {max_len}")

        # Step 2: Duration Predictor
        print("[2/4] Duration predictor...")
        dp_dir = sample_dir / "duration_predictor"
        dp_dir.mkdir(exist_ok=True)

        # Save inputs
        self.save_tensor(dp_dir / "input_text_ids.raw", text_ids)
        self.save_tensor(dp_dir / "input_style_dp.raw", style_dp)
        self.save_tensor(dp_dir / "input_text_mask.raw", text_mask)

        # Run model and save output
        duration_raw = self.tts.duration_predictor.run(None, {
            "text_ids": text_ids,
            "style_dp": style_dp,
            "text_mask": text_mask
        })[0]

        self.save_tensor(dp_dir / "output_duration.raw", duration_raw)
        print(f"  Duration: {duration_raw[0]:.2f}s")

        # Step 3: Text Encoder
        print("[3/4] Text encoder...")
        te_dir = sample_dir / "text_encoder"
        te_dir.mkdir(exist_ok=True)

        # Save inputs
        self.save_tensor(te_dir / "input_text_ids.raw", text_ids)
        self.save_tensor(te_dir / "input_style_ttl.raw", style_ttl)
        self.save_tensor(te_dir / "input_text_mask.raw", text_mask)

        # Run model and save output
        text_emb = self.tts.text_encoder.run(None, {
            "text_ids": text_ids,
            "style_ttl": style_ttl,
            "text_mask": text_mask
        })[0]

        self.save_tensor(te_dir / "output_text_emb.raw", text_emb)
        print(f"  Text embedding: {text_emb.shape}")

        # Step 4: Vector Estimator (multiple steps)
        print("[4/4] Vector estimator (flow matching)...")
        ve_dir = sample_dir / "vector_estimator"
        ve_dir.mkdir(exist_ok=True)

        # Initialize latent noise
        duration = duration_raw / sample_config.get('speed', 1.0)
        wav_length = int(duration[0] * self.tts.sample_rate)
        chunk_size = self.tts.base_chunk_size * self.tts.chunk_compress_factor
        latent_len = (wav_length + chunk_size - 1) // chunk_size
        latent_dim = self.tts.latent_dim * self.tts.chunk_compress_factor

        # Set seed for reproducibility
        np.random.seed(sample_config['seed'])
        noisy_latent = np.random.randn(1, latent_dim, latent_len).astype(np.float32)

        # Create latent mask
        latent_length = np.array([latent_len], dtype=np.int64)
        latent_mask_ids = np.arange(latent_len) < latent_length[:, None]
        latent_mask = latent_mask_ids.astype(np.float32).reshape(1, 1, -1)
        noisy_latent = noisy_latent * latent_mask

        # Pad/truncate latent to fixed length 192 for QNN conversion
        max_latent_len = 192
        original_latent_len = latent_len
        if latent_len > max_latent_len:
            print(f"  WARNING: Truncating latent from {latent_len} to {max_latent_len}")
            noisy_latent = noisy_latent[:, :, :max_latent_len]
            latent_mask = latent_mask[:, :, :max_latent_len]
            latent_len = max_latent_len
        elif latent_len < max_latent_len:
            # Pad noisy_latent with zeros
            padding = np.zeros((1, latent_dim, max_latent_len - latent_len), dtype=noisy_latent.dtype)
            noisy_latent = np.concatenate([noisy_latent, padding], axis=2)
            # Pad latent_mask with zeros
            padding_mask = np.zeros((1, 1, max_latent_len - latent_len), dtype=latent_mask.dtype)
            latent_mask = np.concatenate([latent_mask, padding_mask], axis=2)
            print(f"  Padded latent from {original_latent_len} to {max_latent_len}")
            latent_len = max_latent_len

        # Diffusion loop - save each step
        diffusion_steps = sample_config['steps']
        total_step = np.array([diffusion_steps], dtype=np.float32)

        for step in range(diffusion_steps):
            step_dir = ve_dir / f"step_{step:03d}"
            step_dir.mkdir(exist_ok=True)

            current_step = np.array([step], dtype=np.float32)

            # Save inputs for this step
            self.save_tensor(step_dir / "input_noisy_latent.raw", noisy_latent)
            self.save_tensor(step_dir / "input_text_emb.raw", text_emb)
            self.save_tensor(step_dir / "input_style_ttl.raw", style_ttl)
            self.save_tensor(step_dir / "input_text_mask.raw", text_mask)
            self.save_tensor(step_dir / "input_latent_mask.raw", latent_mask)
            self.save_tensor(step_dir / "input_current_step.raw", current_step)
            self.save_tensor(step_dir / "input_total_step.raw", total_step)

            # Run model
            noisy_latent = self.tts.vector_estimator.run(None, {
                "noisy_latent": noisy_latent,
                "text_emb": text_emb,
                "style_ttl": style_ttl,
                "text_mask": text_mask,
                "latent_mask": latent_mask,
                "current_step": current_step,
                "total_step": total_step
            })[0]

            # Save output
            self.save_tensor(step_dir / "output_latent.raw", noisy_latent)

            if (step + 1) % 5 == 0 or step == diffusion_steps - 1:
                print(f"  Step {step + 1}/{diffusion_steps}")

        # Step 5: Vocoder
        print("[5/5] Vocoder...")
        voc_dir = sample_dir / "vocoder"
        voc_dir.mkdir(exist_ok=True)

        # Save input (final denoised latent)
        self.save_tensor(voc_dir / "input_latent.raw", noisy_latent)

        # Run model and save output
        wav = self.tts.vocoder.run(None, {"latent": noisy_latent})[0]
        wav_trimmed = wav[0, :wav_length]

        self.save_tensor(voc_dir / "output_waveform.raw", wav_trimmed)
        print(f"  Waveform: {len(wav_trimmed)} samples @ {self.tts.sample_rate} Hz")
        print(f"  Duration: {len(wav_trimmed)/self.tts.sample_rate:.2f}s")

        print(f"\n✓ Sample {sample_idx:03d} complete: {sample_dir}")

    def generate_all(self, samples: List[Dict] = None):
        """Generate calibration data for all samples"""

        if samples is None:
            samples = CALIBRATION_SAMPLES

        # Clear existing calibration data
        if self.output_dir.exists():
            print(f"Clearing existing calibration data at {self.output_dir}...")
            shutil.rmtree(self.output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Generate summary
        summary = {
            "num_samples": len(samples),
            "samples": [],
            "model_info": {
                "sample_rate": self.tts.sample_rate,
                "base_chunk_size": self.tts.base_chunk_size,
                "chunk_compress_factor": self.tts.chunk_compress_factor,
                "latent_dim": self.tts.latent_dim,
                "effective_chunk_size": self.tts.base_chunk_size * self.tts.chunk_compress_factor
            }
        }

        # Generate each sample
        for idx, sample_config in enumerate(samples, start=1):
            try:
                self.generate_sample(sample_config, idx)
                summary["samples"].append({
                    "sample_id": f"sample_{idx:03d}",
                    "text": sample_config["text"],
                    "voice": sample_config["voice"],
                    "lang": sample_config["lang"],
                    "category": sample_config["category"],
                    "diffusion_steps": sample_config["steps"],
                    "seed": sample_config["seed"]
                })
            except Exception as e:
                print(f"\n✗ Error generating sample {idx:03d}: {e}")
                import traceback
                traceback.print_exc()
                continue

        # Save summary
        summary_file = self.output_dir / "calibration_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n{'='*80}")
        print(f"✓ Calibration data generation complete!")
        print(f"  Total samples: {len(summary['samples'])}")
        print(f"  Output directory: {self.output_dir}")
        print(f"  Summary: {summary_file}")
        print(f"{'='*80}\n")

        # Print statistics
        print("Sample Statistics:")
        print(f"  Short utterances:  {sum(1 for s in summary['samples'] if 'short' in s['category'])}")
        print(f"  Medium utterances: {sum(1 for s in summary['samples'] if 'medium' in s['category'])}")
        print(f"  Long utterances:   {sum(1 for s in summary['samples'] if 'long' in s['category'])}")
        print(f"  English samples:   {sum(1 for s in summary['samples'] if s['lang'] == 'en')}")
        print(f"  Spanish samples:   {sum(1 for s in summary['samples'] if s['lang'] == 'es')}")
        print(f"  Korean samples:    {sum(1 for s in summary['samples'] if s['lang'] == 'ko')}")
        print(f"  Unique voices:     {len(set(s['voice'] for s in summary['samples']))}")

        return summary


def main():
    parser = argparse.ArgumentParser(
        description="Generate calibration data for Supertonic-2 ONNX models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage Examples:
  # Generate all calibration samples
  python generate_calibration_data.py

  # Custom output directory
  python generate_calibration_data.py --output-dir /path/to/calibration

  # Use custom model directory
  python generate_calibration_data.py --model-dir /path/to/onnx/models

Output Structure:
  calibration_data/
  ├── sample_001/
  │   ├── metadata.json
  │   ├── duration_predictor/ (inputs + output)
  │   ├── text_encoder/ (inputs + output)
  │   ├── vector_estimator/
  │   │   ├── step_000/ (inputs + output)
  │   │   ├── step_001/
  │   │   └── ...
  │   └── vocoder/ (input + output)
  └── calibration_summary.json

Files are saved as:
  - .raw: Binary float32 data
  - .shape.json: Shape and dtype metadata
"""
    )

    parser.add_argument(
        "--model-dir",
        type=str,
        default="model/onnx",
        help="Path to ONNX models directory (default: model/onnx)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="calibration_data",
        help="Output directory for calibration data (default: calibration_data)"
    )

    args = parser.parse_args()

    # Generate calibration data
    generator = CalibrationDataGenerator(
        model_dir=args.model_dir,
        output_dir=args.output_dir
    )

    generator.generate_all()


if __name__ == "__main__":
    main()
