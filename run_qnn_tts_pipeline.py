#!/usr/bin/env python3
"""
Complete QNN TTS Pipeline for QCS6490 HTP Backend

This script runs the entire Supertonic-2 TTS pipeline on Qualcomm QCS6490 HTP:
1. Text preprocessing and input tensor generation
2. Transfer inputs to device
3. Execute all 4 models on HTP backend
4. Retrieve outputs and convert to WAV

Usage:
    python run_qnn_tts_pipeline.py --text "Your text here" --voice F1 --output output.wav
"""

import argparse
import json
import re
import wave
import subprocess
from pathlib import Path
from typing import Tuple
from unicodedata import normalize as unicode_normalize

import numpy as np


class QNNTTSPipeline:
    """Complete TTS pipeline for QNN/HTP deployment"""

    def __init__(self, model_dir: str = "model/onnx", device: str = "root@192.168.31.17", password: str = "oelinux123"):
        """Initialize pipeline"""
        self.model_dir = Path(model_dir)
        self.device = device
        self.password = password
        self.ssh_prefix = f"sshpass -p '{password}' ssh -o StrictHostKeyChecking=no"
        self.scp_prefix = f"sshpass -p '{password}' scp -o StrictHostKeyChecking=no"
        self.deployment_dir = Path("qcs6490_deployment")

        # Load configuration
        with open(self.model_dir / "tts.json") as f:
            self.config = json.load(f)

        self.sample_rate = self.config["ae"]["sample_rate"]
        self.base_chunk_size = self.config["ae"]["base_chunk_size"]
        self.chunk_compress_factor = self.config["ttl"]["chunk_compress_factor"]
        self.latent_dim = self.config["ttl"]["latent_dim"]

        # Load character mapping
        with open(self.model_dir / "unicode_indexer.json") as f:
            self.unicode_indexer = json.load(f)

    def preprocess_text(self, text: str) -> str:
        """Preprocess text (same as ONNX version)"""
        try:
            text = unicode_normalize("NFKD", text)
        except:
            pass

        # Remove emojis
        emoji_pattern = re.compile(
            "[\U0001f600-\U0001f64f\U0001f300-\U0001f5ff\U0001f680-\U0001f6ff"
            "\U0001f700-\U0001f77f\U0001f780-\U0001f7ff\U0001f800-\U0001f8ff"
            "\U0001f900-\U0001f9ff\U0001fa00-\U0001fa6f\U0001fa70-\U0001faff"
            "\u2600-\u26ff\u2700-\u27bf\U0001f1e6-\U0001f1ff]+",
            flags=re.UNICODE
        )
        text = emoji_pattern.sub("", text)

        # Normalize symbols
        replacements = {
            "\u2013": "-", "\u2014": "-", "\u2011": "-",
            "_": " ",
            "\u201c": '"', "\u201d": '"',
            "\u2018": "'", "\u2019": "'",
            "`": "'",
            "[": " ", "]": " ", "|": " ", "/": " ", "#": " "
        }
        for old, new in replacements.items():
            text = text.replace(old, new)

        text = re.sub(r"[\u0302-\u032F]", "", text)
        text = text.replace("@", " at ")
        text = text.replace("e.g.,", "for example,")
        text = text.replace("i.e.,", "that is,")

        for punct in [",", ".", "!", "?", ";", ":", "'"]:
            text = re.sub(f" \\{punct}", punct, text)

        text = re.sub(r'(["\'\`])\1+', r'\1', text)
        text = re.sub(r"\s+", " ", text).strip()

        if not re.search(r"[.!?;:,'\")\]}]$", text):
            text += "."

        return text

    def text_to_ids(self, text: str, lang: str = "en") -> Tuple[np.ndarray, np.ndarray]:
        """Convert text to token IDs (INT32 for QNN)"""
        text = self.preprocess_text(text)
        text = f"<{lang}>{text}</{lang}>"

        ids = []
        for char in text:
            char_code = ord(char)
            if char_code < len(self.unicode_indexer):
                idx = self.unicode_indexer[char_code]
                if idx != -1:
                    ids.append(idx)

        # QNN requires INT32 (not INT64)
        text_ids = np.array([ids], dtype=np.int32)

        length = np.array([len(ids)], dtype=np.int64)
        max_len = length[0]
        mask = np.arange(max_len) < length[:, None]
        text_mask = mask.astype(np.float32).reshape(1, 1, -1)

        return text_ids, text_mask

    def load_voice_style(self, voice_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load voice style embeddings"""
        style_path = self.model_dir.parent / "voice_styles" / f"{voice_name}.json"

        with open(style_path) as f:
            data = json.load(f)

        style_ttl = np.array(data["style_ttl"]["data"], dtype=np.float32)
        style_dp = np.array(data["style_dp"]["data"], dtype=np.float32)

        # Transpose style_dp from [1,8,16] to [1,16,8] as expected by QNN model
        style_dp = np.transpose(style_dp, (0, 2, 1))

        return style_ttl, style_dp

    def quantize_to_ufixed16(self, data: np.ndarray, scale: float, offset: int) -> np.ndarray:
        """Quantize float32 data to UFIXED_POINT_16 format"""
        quantized = np.round(data / scale + offset).astype(np.int16).astype(np.uint16)
        return quantized

    def prepare_inputs(self, text: str, voice_name: str, lang: str, seed: int = None):
        """Prepare all input tensors and save to deployment directory"""
        print(f"\n[1/6] Preparing inputs...")

        # Load voice style
        style_ttl, style_dp = self.load_voice_style(voice_name)

        # Convert text to IDs
        text_ids, text_mask = self.text_to_ids(text, lang)
        print(f"  Text: '{text}'")
        print(f"  Tokens: {text_ids.shape[1]}")

        # Pad and reshape inputs to match QNN model expectations
        # QNN runtime handles quantization internally, so we send float32 data

        # text_ids: INT32 [1, 128]
        text_ids_padded = np.zeros((1, 128), dtype=np.int32)
        text_ids_padded[0, :text_ids.shape[1]] = text_ids[0]

        # text_mask: transpose from [1,1,128] to [1,128,1] and pad to 128
        # Keep as float32 - QNN will quantize internally
        text_mask_transposed = np.zeros((1, 128, 1), dtype=np.float32)
        text_mask_transposed[0, :text_mask.shape[2], 0] = text_mask[0, 0, :]

        # style_dp: already transposed to [1,16,8] in load_voice_style
        # Keep as float32 - QNN will quantize internally

        # Save inputs
        inputs_dir = self.deployment_dir / "inputs"
        inputs_dir.mkdir(parents=True, exist_ok=True)

        text_ids_padded.tofile(inputs_dir / "input_text_ids.raw")
        style_dp.tofile(inputs_dir / "input_style_dp.raw")
        style_ttl.tofile(inputs_dir / "input_style_ttl.raw")
        text_mask_transposed.tofile(inputs_dir / "input_text_mask.raw")

        # Store for later use
        self.text_ids = text_ids
        self.text_mask = text_mask
        self.style_ttl = style_ttl
        self.style_dp = style_dp
        self.seed = seed

        print(f"  ✓ Inputs saved to {inputs_dir}")

    def transfer_to_device(self):
        """Transfer models and inputs to QCS6490 device"""
        print(f"\n[2/6] Transferring to device {self.device}...")

        # Create remote directory structure
        subprocess.run(
            f"{self.ssh_prefix} {self.device} 'mkdir -p /data/qnn_tts/libs /data/qnn_tts/inputs /data/qnn_tts/outputs'",
            shell=True, check=True
        )

        # Transfer model libraries
        subprocess.run(
            f"{self.scp_prefix} qcs6490_deployment/libs/*.so {self.device}:/data/qnn_tts/libs/",
            shell=True, check=True
        )

        # Transfer inputs
        subprocess.run(
            f"{self.scp_prefix} qcs6490_deployment/inputs/*.raw {self.device}:/data/qnn_tts/inputs/",
            shell=True, check=True
        )

        print(f"  ✓ Transfer complete")

    def run_on_device(self, diffusion_steps: int = 10, speed: float = 1.0):
        """Execute all 4 models on HTP backend"""
        print(f"\n[3/6] Running on HTP backend...")

        outputs_dir = self.deployment_dir / "outputs"
        outputs_dir.mkdir(parents=True, exist_ok=True)

        # Model 1: Duration Predictor
        print(f"  [1/4] Duration predictor...")
        cmd = f"""{self.ssh_prefix} {self.device} 'cd /data/qnn_tts && \\
            qnn-net-run \\
                --model libs/libduration_predictor_htp.so \\
                --backend /usr/lib/libQnnHtp.so \\
                --input_list inputs.txt \\
                --output_dir outputs'"""

        # Create input list on device (all inputs on single line, space-separated)
        subprocess.run(
            f"{self.ssh_prefix} {self.device} 'cd /data/qnn_tts && echo \"inputs/input_text_ids.raw inputs/input_style_dp.raw inputs/input_text_mask.raw\" > inputs.txt'",
            shell=True, check=True
        )

        subprocess.run(cmd, shell=True, check=True)

        # Retrieve duration
        subprocess.run(
            f"{self.scp_prefix} {self.device}:/data/qnn_tts/outputs/Result_0/duration.raw {outputs_dir}/",
            shell=True, check=True
        )

        duration_raw = np.fromfile(outputs_dir / "duration.raw", dtype=np.float32)
        duration = duration_raw[0] / speed
        print(f"    Duration: {duration:.2f}s")

        # Model 2: Text Encoder
        print(f"  [2/4] Text encoder...")
        cmd = f"""{self.ssh_prefix} {self.device} 'cd /data/qnn_tts && \\
            qnn-net-run \\
                --model libs/libtext_encoder_htp.so \\
                --backend /usr/lib/libQnnHtp.so \\
                --input_list inputs.txt \\
                --output_dir outputs'"""

        subprocess.run(
            f"{self.ssh_prefix} {self.device} 'cd /data/qnn_tts && echo \"inputs/input_text_ids.raw inputs/input_style_ttl.raw inputs/input_text_mask.raw\" > inputs.txt'",
            shell=True, check=True
        )

        subprocess.run(cmd, shell=True, check=True)

        # Retrieve text embeddings
        subprocess.run(
            f"{self.scp_prefix} {self.device}:/data/qnn_tts/outputs/Result_0/text_emb.raw {outputs_dir}/",
            shell=True, check=True
        )

        text_emb = np.fromfile(outputs_dir / "text_emb.raw", dtype=np.float32).reshape(1, 256, -1)
        print(f"    Text embeddings: {text_emb.shape}")

        # Model 3: Vector Estimator (diffusion loop)
        print(f"  [3/4] Vector estimator ({diffusion_steps} steps)...")

        # Initialize latent noise with FIXED dimensions matching model conversion
        # Model was converted with --input_dim noisy_latent 1,144,192
        latent_dim_fixed = 144  # Fixed latent dimension
        latent_len_fixed = 192  # Fixed latent length

        # Calculate actual dimensions for trimming later
        wav_length = int(duration * self.sample_rate)
        chunk_size = self.base_chunk_size * self.chunk_compress_factor
        latent_len_actual = (wav_length + chunk_size - 1) // chunk_size

        if self.seed is not None:
            np.random.seed(self.seed)

        noisy_latent = np.random.randn(1, latent_dim_fixed, latent_len_fixed).astype(np.float32)

        # Create latent mask using actual length
        latent_mask = np.zeros((1, 1, latent_len_fixed), dtype=np.float32)
        latent_mask[0, 0, :latent_len_actual] = 1.0
        noisy_latent = noisy_latent * latent_mask

        # Diffusion loop
        total_step = np.array([diffusion_steps], dtype=np.float32)

        for step in range(diffusion_steps):
            current_step = np.array([step], dtype=np.float32)

            # Save inputs for this step
            noisy_latent.tofile(self.deployment_dir / "inputs/input_noisy_latent.raw")
            text_emb.tofile(self.deployment_dir / "inputs/input_text_emb.raw")
            latent_mask.tofile(self.deployment_dir / "inputs/input_latent_mask.raw")
            current_step.tofile(self.deployment_dir / "inputs/input_current_step.raw")
            total_step.tofile(self.deployment_dir / "inputs/input_total_step.raw")

            # Transfer inputs
            subprocess.run(
                f"{self.scp_prefix} {self.deployment_dir}/inputs/input_noisy_latent.raw {self.deployment_dir}/inputs/input_text_emb.raw {self.deployment_dir}/inputs/input_latent_mask.raw {self.deployment_dir}/inputs/input_current_step.raw {self.deployment_dir}/inputs/input_total_step.raw {self.device}:/data/qnn_tts/inputs/",
                shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )

            # Run model
            cmd = f"""{self.ssh_prefix} {self.device} 'cd /data/qnn_tts && \\
                qnn-net-run \\
                    --model libs/libvector_estimator_htp.so \\
                    --backend /usr/lib/libQnnHtp.so \\
                    --input_list inputs.txt \\
                    --output_dir outputs' 2>&1 | grep -v 'QnnLog_Level'"""

            subprocess.run(
                f"{self.ssh_prefix} {self.device} 'cd /data/qnn_tts && echo \"inputs/input_noisy_latent.raw inputs/input_text_emb.raw inputs/input_style_ttl.raw inputs/input_latent_mask.raw inputs/input_text_mask.raw inputs/input_current_step.raw inputs/input_total_step.raw\" > inputs.txt'",
                shell=True, check=True
            )

            subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # Retrieve denoised latent
            subprocess.run(
                f"{self.scp_prefix} {self.device}:/data/qnn_tts/outputs/Result_0/denoised_latent.raw {outputs_dir}/ 2>/dev/null",
                shell=True, check=True
            )

            noisy_latent = np.fromfile(outputs_dir / "denoised_latent.raw", dtype=np.float32).reshape(1, latent_dim_fixed, latent_len_fixed)

            if (step + 1) % 5 == 0 or step == diffusion_steps - 1:
                print(f"    Step {step + 1}/{diffusion_steps}")

        # Model 4: Vocoder
        print(f"  [4/4] Vocoder...")

        noisy_latent.tofile(self.deployment_dir / "inputs/input_latent.raw")
        subprocess.run(
            f"{self.scp_prefix} {self.deployment_dir}/inputs/input_latent.raw {self.device}:/data/qnn_tts/inputs/",
            shell=True, check=True
        )

        cmd = f"""{self.ssh_prefix} {self.device} 'cd /data/qnn_tts && \\
            qnn-net-run \\
                --model libs/libvocoder_htp.so \\
                --backend /usr/lib/libQnnHtp.so \\
                --input_list inputs.txt \\
                --output_dir outputs'"""

        subprocess.run(
            f"{self.ssh_prefix} {self.device} 'cd /data/qnn_tts && echo \"inputs/input_latent.raw\" > inputs.txt'",
            shell=True, check=True
        )

        subprocess.run(cmd, shell=True, check=True)

        # Retrieve final audio
        subprocess.run(
            f"{self.scp_prefix} {self.device}:/data/qnn_tts/outputs/Result_0/wav_tts.raw {outputs_dir}/",
            shell=True, check=True
        )

        print(f"  ✓ Inference complete")

        # Load and trim audio
        wav = np.fromfile(outputs_dir / "wav_tts.raw", dtype=np.float32)
        wav_trimmed = wav[:wav_length]

        return wav_trimmed, duration

    def save_wav(self, filename: str, audio: np.ndarray):
        """Save audio as WAV file"""
        print(f"\n[4/6] Saving output...")

        # Normalize and convert to 16-bit PCM
        audio_clipped = np.clip(audio, -1.0, 1.0)
        audio_int16 = (audio_clipped * 32767).astype(np.int16)

        with wave.open(filename, 'w') as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(self.sample_rate)
            f.writeframes(audio_int16.tobytes())

        print(f"  ✓ Saved to: {filename}")
        print(f"  Samples: {len(audio)} @ {self.sample_rate} Hz")
        print(f"  Duration: {len(audio)/self.sample_rate:.2f}s")

    def synthesize(
        self,
        text: str,
        voice_name: str = "F1",
        lang: str = "en",
        diffusion_steps: int = 10,
        speed: float = 1.0,
        seed: int = None,
        output_file: str = "output.wav"
    ):
        """Run complete TTS pipeline on HTP"""
        print(f"\n{'='*70}")
        print(f"QNN TTS Pipeline - HTP Backend")
        print(f"{'='*70}")
        print(f"Device: {self.device}")
        print(f"Voice: {voice_name} | Language: {lang} | Steps: {diffusion_steps}")

        # Prepare inputs
        self.prepare_inputs(text, voice_name, lang, seed)

        # Transfer to device
        self.transfer_to_device()

        # Run inference
        waveform, duration = self.run_on_device(diffusion_steps, speed)

        # Save output
        self.save_wav(output_file, waveform)

        print(f"\n{'='*70}")
        print(f"✓ Pipeline complete")
        print(f"{'='*70}\n")

        return waveform, duration


def main():
    parser = argparse.ArgumentParser(
        description="QNN TTS Pipeline for QCS6490 HTP",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python run_qnn_tts_pipeline.py --text "Hello world" --voice F1

  # With specific seed
  python run_qnn_tts_pipeline.py \\
    --text "Good morning everyone" \\
    --voice M1 \\
    --seed 100

Available voices:
  Female: F1, F2, F3, F4, F5
  Male:   M1, M2, M3, M4, M5
        """
    )

    parser.add_argument(
        "--text", "-t",
        type=str,
        required=True,
        help="Text to synthesize"
    )
    parser.add_argument(
        "--voice", "-v",
        type=str,
        default="F1",
        choices=["F1", "F2", "F3", "F4", "F5", "M1", "M2", "M3", "M4", "M5"],
        help="Voice style (default: F1)"
    )
    parser.add_argument(
        "--lang", "-l",
        type=str,
        default="en",
        choices=["en", "ko", "es", "pt", "fr"],
        help="Language code (default: en)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="board_outputs/qnn_htp_output.wav",
        help="Output WAV file (default: board_outputs/qnn_htp_output.wav)"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=10,
        help="Diffusion steps (default: 10)"
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Speech speed multiplier (default: 1.0)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (default: None = random)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="root@192.168.31.17",
        help="Device SSH address (default: root@192.168.31.17)"
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run pipeline
    pipeline = QNNTTSPipeline(device=args.device)
    pipeline.synthesize(
        text=args.text,
        voice_name=args.voice,
        lang=args.lang,
        diffusion_steps=args.steps,
        speed=args.speed,
        seed=args.seed,
        output_file=args.output
    )


if __name__ == "__main__":
    main()
