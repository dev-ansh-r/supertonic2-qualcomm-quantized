#!/usr/bin/env python3
"""
Supertonic-2 TTS ONNX Inference
Exact replication of official Supertonic library behavior using ONNX models.

This implementation matches the official library's preprocessing, model execution,
and post-processing steps to produce high-quality text-to-speech synthesis.

Usage:
    python supertonic_inference.py --text "Your text here" --voice M1 --output output.wav

Model files required:
    - model/onnx/*.onnx (4 ONNX models)
    - model/onnx/tts.json (configuration)
    - model/onnx/unicode_indexer.json (character mapping)
    - model/voice_styles/*.json (voice embeddings)
"""

import argparse
import json
import re
import wave
from pathlib import Path
from typing import Tuple
from unicodedata import normalize as unicode_normalize

import numpy as np
import onnxruntime as ort


class SupertonicTTS:
    """
    Supertonic-2 Text-to-Speech Engine

    Implements the complete TTS pipeline:
    1. Text preprocessing (normalization, language tagging)
    2. Duration prediction
    3. Text encoding with voice style
    4. Latent diffusion denoising
    5. Vocoding to audio waveform
    """

    def __init__(self, model_dir: str = "model/onnx"):
        """
        Initialize the TTS engine.

        Args:
            model_dir: Path to directory containing ONNX models and configs
        """
        self.model_dir = Path(model_dir)

        # Load configuration
        with open(self.model_dir / "tts.json") as f:
            self.config = json.load(f)

        # Extract audio parameters
        self.sample_rate = self.config["ae"]["sample_rate"]
        self.base_chunk_size = self.config["ae"]["base_chunk_size"]
        self.chunk_compress_factor = self.config["ttl"]["chunk_compress_factor"]
        self.latent_dim = self.config["ttl"]["latent_dim"]

        # Load character-to-index mapping
        with open(self.model_dir / "unicode_indexer.json") as f:
            self.unicode_indexer = json.load(f)

        # Initialize ONNX models
        providers = ["CPUExecutionProvider"]
        self.duration_predictor = ort.InferenceSession(
            str(self.model_dir / "duration_predictor.onnx"), providers=providers
        )
        self.text_encoder = ort.InferenceSession(
            str(self.model_dir / "text_encoder.onnx"), providers=providers
        )
        self.vector_estimator = ort.InferenceSession(
            str(self.model_dir / "vector_estimator.onnx"), providers=providers
        )
        self.vocoder = ort.InferenceSession(
            str(self.model_dir / "vocoder.onnx"), providers=providers
        )

    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text with normalization and cleanup.

        Applies the same preprocessing as official Supertonic library:
        - Unicode normalization (NFKD)
        - Emoji removal
        - Symbol standardization
        - Abbreviation expansion
        - Whitespace cleanup
        - Automatic period addition

        Args:
            text: Raw input text

        Returns:
            Preprocessed text ready for synthesis
        """
        # Unicode normalization
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
            "\u2013": "-", "\u2014": "-", "\u2011": "-",  # Dashes
            "_": " ",  # Underscore to space
            "\u201c": '"', "\u201d": '"',  # Smart quotes
            "\u2018": "'", "\u2019": "'",  # Smart apostrophes
            "`": "'",  # Backtick to apostrophe
            "[": " ", "]": " ", "|": " ", "/": " ", "#": " "  # Special chars to space
        }
        for old, new in replacements.items():
            text = text.replace(old, new)

        # Remove diacritics
        text = re.sub(r"[\u0302-\u032F]", "", text)

        # Expand abbreviations
        text = text.replace("@", " at ")
        text = text.replace("e.g.,", "for example,")
        text = text.replace("i.e.,", "that is,")

        # Fix punctuation spacing
        for punct in [",", ".", "!", "?", ";", ":", "'"]:
            text = re.sub(f" \\{punct}", punct, text)

        # Remove duplicate quotes
        text = re.sub(r'(["\'\`])\1+', r'\1', text)

        # Clean whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # Add period if needed
        if not re.search(r"[.!?;:,'\")\]}]$", text):
            text += "."

        return text

    def text_to_ids(self, text: str, lang: str = "en") -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert text to token IDs and attention mask.

        Args:
            text: Input text (will be preprocessed)
            lang: Language code (en, ko, es, pt, fr)

        Returns:
            Tuple of (text_ids, text_mask) as numpy arrays
        """
        # Preprocess and add language tags
        text = self.preprocess_text(text)
        text = f"<{lang}>{text}</{lang}>"

        # Convert to character indices
        ids = []
        for char in text:
            char_code = ord(char)
            if char_code < len(self.unicode_indexer):
                idx = self.unicode_indexer[char_code]
                if idx != -1:
                    ids.append(idx)

        # Create batch with single text
        text_ids = np.array([ids], dtype=np.int64)

        # Create attention mask
        length = np.array([len(ids)], dtype=np.int64)
        max_len = length[0]
        mask = np.arange(max_len) < length[:, None]
        text_mask = mask.astype(np.float32).reshape(1, 1, -1)

        return text_ids, text_mask

    def load_voice_style(self, voice_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load voice style embeddings from JSON file.

        Args:
            voice_name: Voice style name (F1-F5 for female, M1-M5 for male)

        Returns:
            Tuple of (style_ttl, style_dp) embeddings
        """
        style_path = self.model_dir.parent / "voice_styles" / f"{voice_name}.json"

        with open(style_path) as f:
            data = json.load(f)

        style_ttl = np.array(data["style_ttl"]["data"], dtype=np.float32)
        style_dp = np.array(data["style_dp"]["data"], dtype=np.float32)

        return style_ttl, style_dp

    def synthesize(
        self,
        text: str,
        voice_name: str = "M1",
        lang: str = "en",
        diffusion_steps: int = 10,
        speed: float = 1.0,
        seed: int = None,
        verbose: bool = True
    ) -> Tuple[np.ndarray, float]:
        """
        Synthesize speech from text.

        Args:
            text: Input text to synthesize
            voice_name: Voice style (F1-F5, M1-M5)
            lang: Language code (en, ko, es, pt, fr)
            diffusion_steps: Number of denoising steps (default: 10, more = higher quality)
            speed: Speech speed multiplier (default: 1.0, >1.0 = faster)
            seed: Random seed for reproducibility (default: None = random)
            verbose: Print progress messages

        Returns:
            Tuple of (waveform, duration) where:
                - waveform: Audio samples as float32 array
                - duration: Audio duration in seconds
        """
        if verbose:
            print(f"\n{'='*70}")
            print(f"Supertonic-2 TTS Synthesis")
            print(f"{'='*70}")
            print(f"Text: '{text}'")
            print(f"Voice: {voice_name} | Language: {lang} | Steps: {diffusion_steps}")

        # Load voice style
        style_ttl, style_dp = self.load_voice_style(voice_name)

        # Convert text to IDs
        if verbose:
            print(f"\n[1/5] Text processing...")
        text_ids, text_mask = self.text_to_ids(text, lang)
        if verbose:
            print(f"  Tokens: {text_ids.shape[1]}")

        # Predict duration
        if verbose:
            print(f"[2/5] Predicting duration...")
        duration_raw = self.duration_predictor.run(None, {
            "text_ids": text_ids,
            "style_dp": style_dp,
            "text_mask": text_mask
        })[0]

        duration = duration_raw / speed
        duration_seconds = float(duration[0])
        if verbose:
            print(f"  Duration: {duration_seconds:.2f}s")

        # Encode text with style
        if verbose:
            print(f"[3/5] Encoding text...")
        text_emb = self.text_encoder.run(None, {
            "text_ids": text_ids,
            "style_ttl": style_ttl,
            "text_mask": text_mask
        })[0]

        # Initialize latent noise
        if verbose:
            print(f"[4/5] Diffusion denoising...")

        # Calculate latent dimensions
        wav_length = int(duration[0] * self.sample_rate)
        chunk_size = self.base_chunk_size * self.chunk_compress_factor
        latent_len = (wav_length + chunk_size - 1) // chunk_size
        latent_dim = self.latent_dim * self.chunk_compress_factor

        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)

        # Sample random noise
        noisy_latent = np.random.randn(1, latent_dim, latent_len).astype(np.float32)

        # Create and apply latent mask
        latent_length = np.array([latent_len], dtype=np.int64)
        latent_mask_ids = np.arange(latent_len) < latent_length[:, None]
        latent_mask = latent_mask_ids.astype(np.float32).reshape(1, 1, -1)
        noisy_latent = noisy_latent * latent_mask

        # Diffusion loop
        total_step = np.array([diffusion_steps], dtype=np.float32)
        for step in range(diffusion_steps):
            current_step = np.array([step], dtype=np.float32)

            noisy_latent = self.vector_estimator.run(None, {
                "noisy_latent": noisy_latent,
                "text_emb": text_emb,
                "style_ttl": style_ttl,
                "text_mask": text_mask,
                "latent_mask": latent_mask,
                "current_step": current_step,
                "total_step": total_step
            })[0]

            if verbose and ((step + 1) % 5 == 0 or step == diffusion_steps - 1):
                print(f"  Step {step + 1}/{diffusion_steps}")

        # Vocode to audio
        if verbose:
            print(f"[5/5] Generating audio...")
        wav = self.vocoder.run(None, {"latent": noisy_latent})[0]

        # Trim to exact duration
        wav_trimmed = wav[0, :wav_length]

        if verbose:
            print(f"\n{'='*70}")
            print(f"✓ Synthesis complete")
            print(f"  Samples: {len(wav_trimmed)} @ {self.sample_rate} Hz")
            print(f"  Duration: {len(wav_trimmed)/self.sample_rate:.2f}s")
            print(f"{'='*70}\n")

        return wav_trimmed, duration_seconds


def save_wav(filename: str, audio: np.ndarray, sample_rate: int):
    """
    Save audio waveform as WAV file.

    Args:
        filename: Output file path
        audio: Audio samples as float32 array (range: -1.0 to 1.0)
        sample_rate: Sample rate in Hz
    """
    # Normalize and convert to 16-bit PCM
    audio_clipped = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio_clipped * 32767).astype(np.int16)

    with wave.open(filename, 'w') as f:
        f.setnchannels(1)  # Mono
        f.setsampwidth(2)  # 16-bit
        f.setframerate(sample_rate)
        f.writeframes(audio_int16.tobytes())


def main():
    parser = argparse.ArgumentParser(
        description="Supertonic-2 TTS ONNX Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python supertonic_inference.py --text "Hello world" --voice F1

  # High quality with specific seed
  python supertonic_inference.py \\
    --text "Important announcement" \\
    --voice M1 \\
    --steps 20 \\
    --seed 42

  # Faster speech
  python supertonic_inference.py \\
    --text "Quick message" \\
    --voice F2 \\
    --speed 1.2

Available voices:
  Female: F1, F2, F3, F4, F5
  Male:   M1, M2, M3, M4, M5

Supported languages:
  en (English), ko (Korean), es (Spanish), pt (Portuguese), fr (French)
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
        default="M1",
        choices=["F1", "F2", "F3", "F4", "F5", "M1", "M2", "M3", "M4", "M5"],
        help="Voice style (default: M1)"
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
        default="output/output.wav",
        help="Output WAV file path (default: output/output.wav)"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=10,
        help="Diffusion steps (default: 10, more = better quality)"
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
        help="Random seed for reproducibility (default: None = random)"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="model/onnx",
        help="Path to ONNX models directory (default: model/onnx)"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress messages"
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output).parent
    if output_dir and not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize TTS engine
    tts = SupertonicTTS(model_dir=args.model_dir)

    # Synthesize
    waveform, duration = tts.synthesize(
        text=args.text,
        voice_name=args.voice,
        lang=args.lang,
        diffusion_steps=args.steps,
        speed=args.speed,
        seed=args.seed,
        verbose=not args.quiet
    )

    # Save output
    save_wav(args.output, waveform, tts.sample_rate)

    if not args.quiet:
        print(f"✓ Saved to: {args.output}")


if __name__ == "__main__":
    main()
