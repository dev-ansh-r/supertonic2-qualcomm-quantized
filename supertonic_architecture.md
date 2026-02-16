# Supertonic-2 TTS Architecture — Complete Deep Dive

## For Qualcomm QCS6490 Porting

> This document captures a structured learning session covering the full Supertonic-2 TTS architecture,
> from high-level pipeline down to matrix-level understanding of every component.
> Target platform: Qualcomm QCS6490 (Hexagon HTP V68, SM7325 equivalence)

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [The 4 ONNX Models — Purpose & Execution Order](#2-the-4-onnx-models--purpose--execution-order)
3. [Configuration Files](#3-configuration-files)
4. [Complete Inference Pipeline](#4-complete-inference-pipeline)
5. [Unicode Indexer — The "No G2P" Design](#5-unicode-indexer--the-no-g2p-design)
6. [Inference Script Walkthrough](#6-inference-script-walkthrough)
7. [ConvNeXt Blocks — The Core Building Block](#7-convnext-blocks--the-core-building-block)
8. [Understanding Dims, Kernel Sizes, and Weight Matrices](#8-understanding-dims-kernel-sizes-and-weight-matrices)
9. [Cross-Attention — How Text Becomes Speech](#9-cross-attention--how-text-becomes-speech)
10. [LARoPE — Length-Aware Rotary Position Embedding](#10-larope--length-aware-rotary-position-embedding)
11. [How Convolution Sliding Works](#11-how-convolution-sliding-works)
12. [QCS6490 Conversion Considerations](#12-qcs6490-conversion-considerations)

---

## 1. System Overview

Supertonic-2 is a **66M parameter**, on-device TTS system based on the **Latent Diffusion Model (LDM)** framework. It consists of three core neural components plus a text preprocessing stage, all exported as ONNX models for cross-platform inference.

The system operates at **44.1 kHz** sample rate and uses a **24-dimensional latent space** with temporal compression (factor of 6), making it extremely compact and fast.

Key characteristics:
- **No G2P module** — operates directly on raw Unicode characters
- **66M parameters** total (~305 MB on disk)
- **5 languages** — English, Korean, Spanish, Portuguese, French
- **Up to 167× real-time** on consumer hardware
- **Flow matching** based generation (not autoregressive)

---

## 2. The 4 ONNX Models — Purpose & Execution Order

The inference pipeline runs these models **sequentially**:

```
Text Input
    │
    ▼
┌─────────────────────┐
│  1. TEXT ENCODER     │  (text_encoder.onnx — ~21-27 MB)
│  Encodes text +     │
│  style into text     │
│  representations     │
│  + predicts duration │
└─────────┬───────────┘
          │  text_hidden_states, durations
          ▼
┌─────────────────────┐
│  2. DURATION         │  (duration_predictor.onnx — ~1.5-3.8 MB)
│     PREDICTOR        │
│  Utterance-level     │
│  duration estimation │
│  (prosody/pacing)    │
└─────────┬───────────┘
          │  predicted duration (samples)
          ▼
┌─────────────────────┐
│  3. VECTOR           │  (vector_estimator.onnx — ~132-175 MB)
│     ESTIMATOR        │
│  Flow-matching       │
│  denoiser that maps  │
│  text to latent      │
│  speech (iterative)  │
└─────────┬───────────┘
          │  latent speech representation (24-dim)
          ▼
┌─────────────────────┐
│  4. VOCODER          │  (vocoder.onnx — ~55-101 MB)
│  (Speech Autoencoder │
│   Decoder)           │
│  Decodes latents     │
│  to 44.1kHz waveform │
└─────────┴───────────┘
          │
          ▼
      Audio Output (16-bit WAV, 44100 Hz)
```

### Model 1: Text Encoder (`text_encoder.onnx`, ~21-27 MB)

**Corresponds to:** `ttl.text_encoder` + `ttl.style_encoder` + `ttl.speech_prompted_text_encoder` in tts.json

**What it does:**
- Takes Unicode character indices (from `unicode_indexer.json`) as input
- Embeds characters into 256-dim vectors via a learned character embedding
- Processes through 6-layer ConvNeXt blocks (kernel size 5, intermediate dim 1024)
- Refines with a 4-layer attention encoder (4 heads, 256 hidden, 1024 filter channels)
- Conditions on voice style via a style token attention mechanism (50 style tokens, 2 heads)
- Outputs: **text hidden states** (256-dim per token) used for cross-attention in the vector estimator

**Inputs:** `input_ids` (int64), `attention_mask` (int64), `style` (float32 — the `style_ttl` from voice JSON)
**Outputs:** `last_hidden_state`, `raw_durations`

### Model 2: Duration Predictor (`duration_predictor.onnx`, ~1.5-3.8 MB)

**Corresponds to:** `dp` section in tts.json

**What it does:**
- Predicts the **utterance-level duration** — how long (in audio samples) the entire utterance should be
- This is NOT phoneme-level duration; it works at the sentence level, making the architecture much simpler
- Uses its own smaller text encoder (64-dim embeddings, 6-layer ConvNeXt, 2-layer attention)
- Has its own style encoder with 8 style tokens (vs 50 in the TTL)
- A small MLP predictor (2 layers, 128 hidden) combines sentence and style features

**Inputs:** Text indices, `style_dp` from voice JSON
**Outputs:** Duration in samples → converted to latent length via: `latent_length = ceil(duration / (base_chunk_size × chunk_compress_factor))` = `ceil(duration / 3072)`

### Model 3: Vector Estimator (`vector_estimator.onnx`, ~132-175 MB) — **LARGEST MODEL**

**Corresponds to:** `ttl.vector_field` + `ttl.flow_matching` in tts.json

**What it does:**
- This is the **flow-matching denoiser** — the heart of the TTS system
- It iteratively transforms random Gaussian noise into a clean latent speech representation
- Uses a **conditional flow matching** approach (similar to diffusion, but more efficient)
- The vector field network has:
  - Input projection: latent (24-dim × 6 compressed) → 512-dim
  - 4 main blocks, each containing:
    - Time conditioning (64-dim timestep encoding)
    - Style conditioning (256-dim from voice style)
    - **Cross-attention** with text encoder output (4 heads, with LARoPE)
    - ConvNeXt processing blocks with dilations [1,2,4,8]
  - Final ConvNeXt block (4 layers) + output projection back to 24-dim latent

**Inputs:** `latent` (noise or partially denoised), `latent_mask`, `timestep`, `text_hidden_state`, `style`
**Outputs:** Updated latent (directly, with Euler integration baked in)

**Called N times** per inference (N = `total_steps`, default 5, can be as low as 2 for speed)

### Model 4: Vocoder (`vocoder.onnx`, ~55-101 MB)

**Corresponds to:** `ae.decoder` in tts.json

**What it does:**
- Decodes the 24-dimensional latent representation back into a raw audio waveform
- Uses a 10-layer ConvNeXt decoder with dilations [1,2,4,1,2,4,1,1,1,1]
- Final head: 512→2048→512 with kernel size 3
- Produces audio at **44,100 Hz** sample rate
- Each latent frame corresponds to 512 audio samples (base_chunk_size)

**Inputs:** Denoised latent tensor (24-dim)
**Outputs:** Raw audio waveform (float32, 44.1kHz)

---

## 3. Configuration Files

### `tts.json` — Model Architecture Configuration

This file defines **every hyperparameter** of the neural architecture. It is critical for:
- Knowing tensor shapes and dimensions for ONNX model I/O
- Understanding the latent space geometry (24-dim, compression factor 6)
- Configuring any custom runtime or conversion tools
- Verifying model compatibility during conversion

**Key constants from tts.json:**

| Parameter | Value | Where Used |
|-----------|-------|------------|
| `latent_dim` | 24 | All models — the dimensionality of the latent audio space |
| `chunk_compress_factor` | 6 | Temporal compression — 6 latent frames become 1 |
| `base_chunk_size` (ae) | 512 | Each latent frame = 512 audio samples |
| `sample_rate` | 44100 | Output audio rate |
| `n_fft` / `win_length` | 2048 | Mel spectrogram params (encoder only, not needed at inference) |
| `hop_length` | 512 | Matches base_chunk_size |
| `n_mels` | 228 | Mel channels (encoder only) |
| `sig_min` | 0 | Flow matching noise floor |
| `normalizer.scale` (ttl) | 0.25 | Latent normalization scale |
| `normalizer.scale` (dp) | 1.0 | Duration predictor normalization |

**Effective latent dimensions:**
- Input to vector field: `24 × 6 = 144` channels (latent_dim × chunk_compress_factor)
- Temporal resolution: 1 latent frame = 512 × 6 = **3072 audio samples** ≈ 69.7ms at 44.1kHz

### `unicode_indexer.json` — Character-to-Index Mapping

This is the **tokenizer** for Supertonic — a simple lookup table, not a neural model.

- Maps every supported Unicode character to a unique integer index
- These indices become the `input_ids` fed into the text encoder and duration predictor
- Covers characters across all 5 supported languages (en, ko, es, pt, fr)
- Includes special tokens for language tags (`<en>`, `</en>`, `<ko>`, etc.)
- Size: ~262 KB (covers thousands of Unicode codepoints)
- **No conversion needed** — implement as preprocessing in application code

---

## 4. Complete Inference Pipeline

Here is the exact step-by-step execution flow:

```
STEP 1: Text Preprocessing (CPU, no neural model)
├── Normalize text (unicode normalization, emoji removal, symbol replacement)
├── For multilingual: wrap text in language tags: "<en>Hello world</en>"
├── Chunk long text into segments (300 chars for en/es/pt/fr, 120 for ko)
├── Convert characters to indices using unicode_indexer.json
└── Create attention mask

STEP 2: Load Voice Style (CPU, no neural model)
├── Load voice JSON file (e.g., M1.json, F2.json)
├── Extract style_ttl (50 × 256 float32 tensor — for text encoder + vector estimator)
└── Extract style_dp (8 × 16 float32 tensor — for duration predictor)

STEP 3: Duration Prediction (duration_predictor.onnx — RUN ONCE)
├── Input: text indices + style_dp
├── Output: predicted duration in samples
├── Compute latent_length = ceil(duration / speed / 3072)
└── Create latent_mask (binary mask for valid positions)

STEP 4: Text Encoding (text_encoder.onnx — RUN ONCE)
├── Input: text indices + attention_mask + style_ttl
└── Output: text_hidden_states (seq_len × 256)

STEP 5: Initialize Random Latent (CPU)
├── Shape: [1, 144, latent_length]  (144 = 24 × 6)
├── Sample from standard normal distribution
└── Apply latent_mask (zero out invalid positions)

STEP 6: Flow Matching Denoising Loop (vector_estimator.onnx — RUN N TIMES)
├── For step = 0 to total_steps-1:
│   ├── t = step / total_steps
│   ├── Run vector_estimator(latent, latent_mask, t, text_hidden, style_ttl)
│   ├── Get updated latent (Euler step baked into model)
│   └── Feed output back as input for next step
└── Output: denoised latent representation

STEP 7: Vocoding (vocoder.onnx — RUN ONCE)
├── Input: denoised latent
├── Output: raw audio waveform (float32, 44.1kHz)
└── Trim to actual duration using latent_mask

STEP 8: Post-processing (CPU)
├── Apply speed adjustment (time-stretching)
├── Concatenate chunks with silence gaps (0.3s default)
└── Convert to 16-bit PCM WAV
```

---

## 5. Unicode Indexer — The "No G2P" Design

### What is `unicode_indexer.json`?

It's a dictionary that maps every raw Unicode character to a unique integer ID. Think of it as a simple lookup table:

```json
{
  "a": 1,
  "b": 2,
  "c": 3,
  "가": 500,
  "나": 501,
  "é": 1200,
  "ñ": 1201,
  "<en>": 5000,
  "</en>": 5001,
  "<ko>": 5002
}
```

(Illustrative values — the real file covers thousands of codepoints across 5 languages.)

### Why This Matters — No G2P

Most TTS systems work like this:

```
Traditional TTS:
  "Hello" → G2P module → /h ɛ l oʊ/ (phonemes) → phoneme IDs → model
```

Supertonic **skips this entirely**:

```
Supertonic:
  "Hello" → unicode_indexer.json → [34, 12, 55, 55, 72] → model
```

There's no grapheme-to-phoneme conversion, no pronunciation dictionary, no external linguistic tool. The model learns to map raw characters directly to speech.

### How It Feeds Into the Text Encoder

**Step 1: Preprocessing (application code, not neural)**

```
Input text: "Hello world"
For multilingual: "<en>Hello world</en>"

Character-by-character lookup using unicode_indexer.json:
  '<' → idx_1
  'e' → idx_2
  'n' → idx_3
  '>' → idx_4
  'H' → idx_5
  'e' → idx_6
  'l' → idx_7
  'l' → idx_7  (same character = same index)
  'o' → idx_8
  ...

Result: input_ids = [idx_1, idx_2, idx_3, ..., idx_N]
Also create: attention_mask = [1, 1, 1, ..., 1, 0, 0]
```

**Step 2: Inside text_encoder.onnx**

```
input_ids (integer indices)
    │
    ▼
┌──────────────────────────────┐
│  Character Embedding Layer   │  char_emb_dim: 256
│  Each index → 256-dim vector │
│  (learned embedding table)   │
└──────────┬───────────────────┘
           │  Shape: [seq_len, 256]
           ▼
┌──────────────────────────────┐
│  6× ConvNeXt Blocks          │  Captures local character
│  (kernel=5, 1024 intermediate)│  patterns and context
└──────────┬───────────────────┘
           │  Shape: [seq_len, 256]
           ▼
┌──────────────────────────────┐
│  4× Attention Encoder Layers │  Self-attention lets each
│  (256 hidden, 4 heads)       │  character attend to all others
└──────────┬───────────────────┘
           │  Shape: [seq_len, 256]
           ▼
┌──────────────────────────────┐
│  Style Conditioning          │  Voice style tokens cross-
│  (50 tokens, 256-dim, 2 heads)│  attended with text
└──────────┬───────────────────┘
           │
           ▼
     text_hidden_states [seq_len, 256]
```

The **character embedding layer** contains a matrix of shape `[vocab_size, 256]`. Each integer index selects one row from this matrix, producing a 256-dimensional vector.

### Why the Model Can Work Without G2P

The ConvNeXt blocks with kernel size 5 act as learned "character n-gram" processors. By looking at windows of 5 characters at a time, the model learns patterns like:
- "th" → the "th" sound
- "ough" → various English pronunciations
- "tion" → the "shun" sound
- "가" → Korean syllable block (already encodes onset+vowel+coda)

The subsequent self-attention layers resolve ambiguities by looking at full sentence context — for example, "read" (present) vs "read" (past tense).

### Implementation for QCS6490

The unicode indexer is pure preprocessing logic:
1. Load `unicode_indexer.json` into a hashmap in your C++ application code
2. For each character in input text, look up its integer index
3. Pack indices into an int64 tensor
4. Create attention mask tensor
5. Feed both into `text_encoder.onnx`

No neural computation, no special hardware needed.

The actual structure is an **array indexed by Unicode codepoint** (not a key-value dict):
`unicode_indexer[72]` gives the model's internal ID for character `'H'` (codepoint 72). Characters mapping to `-1` are dropped.

---

## 6. Inference Script Walkthrough

### Initialization (`__init__`)

```python
self.sample_rate = self.config["ae"]["sample_rate"]           # 44100
self.base_chunk_size = self.config["ae"]["base_chunk_size"]   # 512
self.chunk_compress_factor = self.config["ttl"]["chunk_compress_factor"]  # 6
self.latent_dim = self.config["ttl"]["latent_dim"]            # 24
```

These four values from `tts.json` define the entire geometry of the latent space:

**1 latent frame = 512 × 6 = 3,072 audio samples = ~69.7ms of audio at 44.1kHz**

Then the 4 ONNX models are loaded into memory. Nothing runs yet.

### Step 1: Load Voice Style

```python
style_ttl, style_dp = self.load_voice_style(voice_name)
```

Opens something like `M1.json` and extracts two pre-computed tensors:

| Tensor | Shape | Used By | What It Represents |
|--------|-------|---------|-------------------|
| `style_ttl` | `[50, 256]` | text_encoder + vector_estimator | 50 style tokens, each 256-dim — captures timbre, pitch, speaking characteristics |
| `style_dp` | `[8, 16]` | duration_predictor | 8 style tokens, each 16-dim — captures pacing/rhythm preferences |

These were extracted from reference audio during voice building. At inference, they're just loaded as constants.

### Step 2: Text → Integer IDs (unicode_indexer in action)

```python
text_ids, text_mask = self.text_to_ids(text, lang)
```

Inside `text_to_ids`:

```python
# Preprocess (normalize unicode, remove emoji, fix symbols)
text = self.preprocess_text(text)

# Wrap with language tags
text = f"<{lang}>{text}</{lang}>"
# Example: "<en>Hello world.</en>"

# Character-by-character lookup
ids = []
for char in text:
    char_code = ord(char)                        # 'H' → 72
    if char_code < len(self.unicode_indexer):
        idx = self.unicode_indexer[char_code]     # 72 → some model-internal index
        if idx != -1:                             # -1 means "unknown, skip"
            ids.append(idx)

text_ids = np.array([ids], dtype=np.int64)     # Shape: [1, seq_len]
text_mask = ...                                 # Shape: [1, 1, seq_len], all 1.0s
```

The mask is `[1, 1, seq_len]` (with a middle dimension) because it will be broadcast against attention matrices later inside the models.

**At this point, no neural model has run yet. Everything so far is pure preprocessing on CPU.**

### Step 3: Duration Prediction — `duration_predictor.onnx` (1st model)

```python
duration_raw = self.duration_predictor.run(None, {
    "text_ids": text_ids,        # [1, seq_len] int64
    "style_dp": style_dp,        # [8, 16] float32
    "text_mask": text_mask        # [1, 1, seq_len] float32
})[0]

duration = duration_raw / speed
```

**What happens inside the model (from tts.json `dp` section):**

```
text_ids [1, seq_len]
    │
    ▼
Character Embedding (64-dim)          ← dp.sentence_encoder.char_emb_dim: 64
    │
    ▼
6× ConvNeXt (ksz=5, 256 intermediate) ← dp.sentence_encoder.convnext
    │
    ▼
2× Self-Attention (2 heads, 64 hidden) ← dp.sentence_encoder.attn_encoder
    │
    ▼
Projection → 64-dim                    ← dp.sentence_encoder.proj_out
    │
    ▼
    ├──── sentence representation (pooled)
    │
    │     style_dp [8, 16]
    │         │
    │         ▼
    │     Style Encoder:
    │       4× ConvNeXt → Style Token Layer (8 tokens, 2 heads)
    │         │
    │         ▼
    │     style representation
    │
    ▼─────────┘
Predictor MLP (128 hidden, 2 layers)   ← dp.predictor
    │
    ▼
Single scalar: predicted duration in seconds
```

**Utterance-level** prediction, not per-phoneme. That's what makes it so small (~1.5 MB).

### Step 4: Text Encoding — `text_encoder.onnx` (2nd model)

```python
text_emb = self.text_encoder.run(None, {
    "text_ids": text_ids,        # [1, seq_len] int64
    "style_ttl": style_ttl,      # [50, 256] float32
    "text_mask": text_mask        # [1, 1, seq_len] float32
})[0]
```

**What happens inside (from tts.json `ttl` section):**

```
text_ids [1, seq_len]
    │
    ▼
Character Embedding (256-dim)           ← ttl.text_encoder.char_emb_dim: 256
    │
    ▼
6× ConvNeXt (ksz=5, 1024 intermediate) ← ttl.text_encoder.convnext
    │                                      Learns "character n-grams"
    ▼
4× Self-Attention (4 heads, 256 hidden) ← ttl.text_encoder.attn_encoder
    │                                      Resolves full-sentence context
    ▼
Projection → 256-dim                    ← ttl.text_encoder.proj_out
    │
    │     style_ttl [50, 256]
    │         │
    │         ▼
    │     Style Encoder:
    │       proj_in (24×6 → 256)
    │       6× ConvNeXt
    │       Style Token Layer (50 tokens, 256-dim keys/values, 2 heads)
    │         │
    │         ▼
    │     style conditioning
    │
    ▼─────────┘
Speech-Prompted Text Encoder:           ← ttl.speech_prompted_text_encoder
  Cross-attention (2 heads, 256-dim)
  Text attends to style → style-conditioned text
    │
    ▼
text_emb [1, seq_len, 256]   ← "what to say" + "how to say it"
```

The same text "Hello" produces different embeddings for voice M1 vs F2, because the style information is baked in.

### Step 5: Initialize Random Latent (CPU, no model)

```python
wav_length = int(duration[0] * self.sample_rate)          # e.g., 2.5s × 44100 = 110250 samples
chunk_size = self.base_chunk_size * self.chunk_compress_factor  # 512 × 6 = 3072
latent_len = (wav_length + chunk_size - 1) // chunk_size  # ceil division → e.g., 36
latent_dim = self.latent_dim * self.chunk_compress_factor  # 24 × 6 = 144

noisy_latent = np.random.randn(1, latent_dim, latent_len).astype(np.float32)
# Shape: [1, 144, 36]  ← pure Gaussian noise
```

Shape `[1, 144, latent_len]` means:
- **144 channels** = 24-dim latent × 6 compression factor (latent is "unrolled" along channels)
- **latent_len** = number of temporal frames, each representing 3,072 audio samples (~70ms)

```python
latent_mask = ...  # Shape: [1, 1, latent_len], 1.0 for valid positions
noisy_latent = noisy_latent * latent_mask  # Zero out padding
```

### Step 6: Flow Matching Loop — `vector_estimator.onnx` (3rd model, runs N times)

```python
total_step = np.array([diffusion_steps], dtype=np.float32)  # e.g., [10]

for step in range(diffusion_steps):
    current_step = np.array([step], dtype=np.float32)       # [0], [1], ..., [9]

    noisy_latent = self.vector_estimator.run(None, {
        "noisy_latent": noisy_latent,    # [1, 144, latent_len]
        "text_emb": text_emb,            # [1, seq_len, 256]
        "style_ttl": style_ttl,          # [50, 256]
        "text_mask": text_mask,           # [1, 1, seq_len]
        "latent_mask": latent_mask,       # [1, 1, latent_len]
        "current_step": current_step,     # scalar
        "total_step": total_step          # scalar
    })[0]
```

**What happens inside each call (from tts.json `ttl.vector_field`):**

```
noisy_latent [1, 144, T]
    │
    ▼
proj_in: 144 → 512                      ← ttl.vector_field.proj_in
    │
    │   current_step/total_step
    │       │
    │       ▼
    │   Time Encoder → 64-dim            ← ttl.vector_field.time_encoder
    │
    │   Repeat 4× Main Blocks:           ← ttl.vector_field.main_blocks (n_blocks: 4)
    │   ┌─────────────────────────────────────────────┐
    │   │  Time Conditioning (512 + 64-dim)           │
    │   │  Style Conditioning (512 + 256-dim)         │
    │   │  Text Cross-Attention:                      │
    │   │    latent queries × text_emb keys/values    │
    │   │    4 heads, with LARoPE (base=10000, ×10)   │
    │   │    ↑ THIS is where text→speech alignment    │
    │   │      happens automatically                   │
    │   │  ConvNeXt_0: 4 layers, dilations [1,2,4,8] │
    │   │  ConvNeXt_1: 1 layer, dilation [1]          │
    │   │  ConvNeXt_2: 1 layer, dilation [1]          │
    │   └─────────────────────────────────────────────┘
    │
    ▼
Last ConvNeXt: 4 layers, dilations [1,1,1,1]
    │
    ▼
proj_out: 512 → 144                     ← ttl.vector_field.proj_out
    │
    ▼
Updated noisy_latent [1, 144, T]
```

**Critical detail:** The model **returns the updated latent directly** — the Euler integration step is **baked inside the ONNX model**. The model takes `current_step` and `total_step` as inputs and handles the update internally.

### Step 7: Vocoding — `vocoder.onnx` (4th model, runs once)

```python
wav = self.vocoder.run(None, {"latent": noisy_latent})[0]
wav_trimmed = wav[0, :wav_length]
```

**What happens inside (from tts.json `ae.decoder`):**

```
denoised latent [1, 144, T]
    │
    ▼ (reshape internally to [1, 24, T×6])
    │
Initial ConvNeXt (ksz=7)
    │
    ▼
10× ConvNeXt Layers (ksz=7, 512 hidden, 2048 intermediate)
  dilations: [1,2,4,1,2,4,1,1,1,1]
    │
    ▼
Head: 512 → 2048 → 512 (ksz=3)
    │
    ▼
Raw waveform [1, num_samples]  at 44100 Hz
```

### Visual Summary of Full Code Flow

```
"Hello world"
      │
      ▼
preprocess_text()          ─── CPU: normalize, cleanup → "Hello world."
      │
      ▼
text_to_ids()              ─── CPU: unicode_indexer lookup
      │                         "<en>Hello world.</en>" → [43, 8, 65, ...]
      │
      ├─── text_ids [1, seq_len]
      │    text_mask [1, 1, seq_len]
      │
      ├────────────────────────┐
      │                        │
      ▼                        ▼
┌─────────────┐    ┌───────────────────┐
│  DURATION    │    │  TEXT ENCODER      │
│  PREDICTOR   │    │  .onnx            │
│  .onnx       │    │                   │
│ +style_dp    │    │ +style_ttl        │
│ → duration   │    │ → text_emb        │
│   (scalar)   │    │   [1, seq, 256]   │
└──────┬───────┘    └────────┬──────────┘
       │                     │
       ▼                     │
  Calculate shapes:          │
  wav_length = dur × 44100   │
  latent_len = ⌈wav/3072⌉   │
  Initialize noise           │
  [1, 144, latent_len]       │
       │                     │
       ▼                     │
┌──────────────────────────────────────┐
│         VECTOR ESTIMATOR .onnx       │
│         (runs N times)               │
│                                      │
│  noise ──step 0──► slightly clean    │
│        ──step 1──► cleaner           │◄──┘
│        ──step N──► clean latent      │
└──────────────────┬───────────────────┘
                   │
                   ▼
         ┌─────────────────┐
         │  VOCODER .onnx  │
         │  latent → audio │
         └────────┬────────┘
                  │
                  ▼
            wav_trimmed → 16-bit PCM .wav
```

---

## 7. ConvNeXt Blocks — The Core Building Block

### What Is a ConvNeXt Block?

ConvNeXt (2022) is a "modernized ConvNet" — standard convolution upgraded with design tricks from Transformers, while remaining pure convolution. Matches Transformer performance but runs significantly faster and more efficiently.

A single ConvNeXt block:

```
Input x [B, C, T]
    │
    ▼
┌─────────────────────────────────┐
│  Depthwise Conv1D               │  kernel_size=5, groups=C
│  (each channel convolved        │  (C independent filters,
│   independently)                │   NOT C×C cross-channel filters)
└──────────┬──────────────────────┘
           │
           ▼
┌─────────────────────────────────┐
│  Layer Normalization            │  (borrowed from Transformers)
└──────────┬──────────────────────┘
           │
           ▼
┌─────────────────────────────────┐
│  Pointwise Conv1D (1×1)        │  C → intermediate_dim (expansion)
│  (Linear projection up)        │  e.g., 256 → 1024 or 512 → 1024
└──────────┬──────────────────────┘
           │
           ▼
┌─────────────────────────────────┐
│  GELU Activation                │  (smooth ReLU, from Transformers)
└──────────┬──────────────────────┘
           │
           ▼
┌─────────────────────────────────┐
│  Pointwise Conv1D (1×1)        │  intermediate_dim → C (compression)
│  (Linear projection down)      │  e.g., 1024 → 256 or 1024 → 512
└──────────┬──────────────────────┘
           │
           ▼
      output + x  ← Residual connection (add input back)
```

### Three Key Design Principles

**1. Depthwise separable convolution**

Normal Conv1D with 512 channels and kernel 5: 512×512×5 = **1,310,720** parameters.
Depthwise: 512×5 = **2,560** parameters. That's **512× fewer** for spatial mixing.

**2. Inverted bottleneck**

Channel dimension expands (256→1024) then contracts (1024→256). Cross-channel mixing happens in the expanded space with more capacity.

**3. Minimal activations/normalizations**

Only one GELU and one LayerNorm per block (vs BatchNorm+ReLU after every conv in traditional ConvNets).

### Why ConvNeXt and Not Transformers?

```
Transformer Self-Attention:
  Complexity: O(T² × C)    ← quadratic in sequence length
  Memory:     O(T²)        ← stores full attention matrix
  Great at:   global context, long-range dependencies
  Problem:    slow for long sequences, memory hungry

ConvNeXt:
  Complexity: O(T × C × K) ← linear in sequence length
  Memory:     O(T × C)     ← no attention matrix
  Great at:   local patterns, fixed receptive field, parallelism
  Problem:    limited receptive field per layer (only K positions)
```

For TTS, speech has strong **local structure** — pronunciation depends mostly on nearby characters, and audio features are highly correlated with temporal neighbors.

### How Supertonic Solves the Receptive Field Problem

**Strategy 1: Stack many layers.** With 6 layers and kernel 5, effective receptive field ≈ 25 characters.

**Strategy 2: Dilated convolutions** in the vector estimator:

```json
"convnext_0": {
    "num_layers": 4,
    "dilation_lst": [1, 2, 4, 8]
}
```

```
Dilation = 1 (normal):    [x x x x x]          sees 5 consecutive positions
Dilation = 2:             [x . x . x . x . x]  sees 5 positions spanning 9
Dilation = 4:             [x . . . x . . . x . . . x . . . x]  spans 17
Dilation = 8:             spans 33 positions
```

After layers with dilations [1,2,4,8], the receptive field covers the entire latent sequence without attention.

**Strategy 3: Use attention only where essential.** Self-attention in text encoder (4 layers) and cross-attention in vector estimator. Heavy lifting is all ConvNeXt.

### Where ConvNeXt Appears in Every Model

**Duration Predictor (~1.5 MB):**
```
Sentence Encoder: 6× ConvNeXt  idim=64, ksz=5, intermediate=256
Style Encoder:    4× ConvNeXt  idim=64, ksz=5, intermediate=256
```

**Text Encoder (~21-27 MB):**
```
Text Encoder:  6× ConvNeXt  idim=256, ksz=5, intermediate=1024
Style Encoder: 6× ConvNeXt  idim=256, ksz=5, intermediate=1024
```

**Vector Estimator (~132-175 MB) — Most Critical:**
```
4× Main Blocks, each containing:
  convnext_0:  4 layers, idim=512, intermediate=1024, dilations=[1,2,4,8]
  convnext_1:  1 layer,  idim=512, intermediate=1024, dilation=[1]
  convnext_2:  1 layer,  idim=512, intermediate=1024, dilation=[1]
Final:
  last_convnext: 4 layers, idim=512, intermediate=1024, dilations=[1,1,1,1]

Total: 28 ConvNeXt layers — the vast majority of the 66M parameters
```

**Vocoder (~55-101 MB):**
```
Decoder: 10× ConvNeXt  idim=512, ksz=7, intermediate=2048, dilations=[1,2,4,1,2,4,1,1,1,1]
```

### Parameter Efficiency

```
Single ConvNeXt block (idim=512, intermediate=1024, ksz=5):
  Depthwise Conv:     512 × 5          =     2,560 params
  LayerNorm:          512 × 2          =     1,024 params
  Pointwise Up:       512 × 1024       =   524,288 params
  Pointwise Down:     1024 × 512       =   524,288 params
  Total:                                ≈ 1,052,160 params (~1M)

Equivalent Transformer layer (dim=512, 4 heads):
  Q,K,V,O projections: 4 × 512 × 512  = 1,048,576 params
  FFN up+down:     512×2048 + 2048×512 = 2,097,152 params
  Total:                                ≈ 3,147,776 params (~3.1M)
```

ConvNeXt is **~3× cheaper in parameters** and has no O(T²) attention computation.

### The Big Picture

```
Without ConvNeXt (pure Transformer):    With ConvNeXt (Supertonic):
  ~200M+ parameters                       66M parameters
  O(T²) at every layer                    O(T) convolutions mostly
  Needs GPU for real-time                  167× real-time on M4 Pro CPU
  Too large for edge devices               Fits on QCS6490 with 16GB RAM
```

---

## 8. Understanding Dims, Kernel Sizes, and Weight Matrices

When we say "dim=256" or "kernel_size=5", we're describing the **shape of weight matrices** — actual arrays of floating point numbers stored inside the ONNX model file. These aren't abstract settings. They are the learned parameters.

### A Linear Layer (dim_in → dim_out)

When tts.json says `"idim": 256, "odim": 256`, there is literally a matrix stored in the model:

```
Weight matrix W: shape [256, 256]  ← 65,536 float32 numbers
Bias vector b:   shape [256]       ← 256 float32 numbers

Operation:  output = input × W + b

    input          ×        W              +    b        =    output
  [1, T, 256]          [256, 256]            [256]         [1, T, 256]
                         ↑
                    This is literally a file
                    of 65,536 floats sitting
                    inside the .onnx file
```

Every "projection" in tts.json (`proj_in`, `proj_out`, etc.) is one of these matrices.

### A 1D Convolution (kernel_size=5, channels=256)

When tts.json says `"ksz": 5, "idim": 256`, there's a 3D weight tensor:

```
For a NORMAL Conv1D (256 in, 256 out, kernel 5):

  Weight: shape [256, 256, 5]  ← 327,680 floats
                  ↑    ↑    ↑
                  │    │    └── kernel_size: how many time positions to look at
                  │    └─────── input channels
                  └──────────── output channels

What it does physically:

  Input signal:  [..., x₁, x₂, x₃, x₄, x₅, ...]    (each xᵢ is a 256-dim vector)
                          ↓    ↓    ↓    ↓    ↓
  Kernel window:        [w₁,  w₂,  w₃,  w₄,  w₅]    (each wᵢ is a 256×256 matrix)
                          ↓    ↓    ↓    ↓    ↓
  Output at this position = x₁·w₁ + x₂·w₂ + x₃·w₃ + x₄·w₄ + x₅·w₅
                          = one 256-dim vector

  Then slide the window one position right and repeat.
```

So `kernel_size=5` means the convolution looks at **5 consecutive positions** at a time.

### Depthwise Conv1D (what ConvNeXt uses)

```
For DEPTHWISE Conv1D (256 channels, kernel 5):

  Weight: shape [256, 1, 5]  ← only 1,280 floats! (not 327,680)
                  ↑   ↑  ↑
                  │   │  └── kernel_size
                  │   └───── each channel only has 1 filter (no cross-channel)
                  └────────── number of channels

  Channel 0:   has its own 5 weights, operates ONLY on channel 0 of input
  Channel 1:   has its own 5 weights, operates ONLY on channel 1 of input
  ...
  Channel 255: has its own 5 weights, operates ONLY on channel 255
```

### Intermediate Dim (the expansion)

When tts.json says `"intermediate_dim": 1024` with `"idim": 256`:

```
Pointwise Up:    W₁ shape [256, 1024]   ← 262,144 floats
Pointwise Down:  W₂ shape [1024, 256]   ← 262,144 floats

Input [T, 256]  →  × W₁  →  [T, 1024]  →  GELU  →  × W₂  →  [T, 256]
                      expand              activation    compress
                      wider space         (nonlinearity) back down
```

### How This Maps to QNN Conversion

When you run ONNX model inspection or QNN conversion tools, you'll see these exact shapes:

```
text_encoder.convnext.0.dwconv.weight    [256, 1, 5]     ← depthwise conv kernel
text_encoder.convnext.0.pwconv1.weight   [1024, 256]     ← pointwise up
text_encoder.convnext.0.pwconv2.weight   [256, 1024]     ← pointwise down
text_encoder.convnext.0.norm.weight      [256]           ← layernorm scale
text_encoder.convnext.0.norm.bias        [256]           ← layernorm bias
```

Every number in tts.json directly corresponds to a dimension of a real weight tensor inside the ONNX file.

---

## 9. Cross-Attention — How Text Becomes Speech

### Regular Self-Attention

A sequence attends to **itself**:

```
Input: X [1, T, 256]  ← T tokens, each 256-dim

Three weight matrices (stored in the model):
  Wq [256, 256]  ← Query projection
  Wk [256, 256]  ← Key projection
  Wv [256, 256]  ← Value projection

Q = X × Wq  →  [1, T, 256]    "What am I looking for?"
K = X × Wk  →  [1, T, 256]    "What do I contain?"
V = X × Wv  →  [1, T, 256]    "What information do I carry?"

Attention scores = Q × Kᵀ  →  [1, T, T]   ← every position vs every other
                                               THIS is the O(T²) cost

Attention weights = softmax(scores / √256)  →  [1, T, T]

Output = weights × V  →  [1, T, 256]
```

### Cross-Attention (two different sequences)

In Supertonic's vector estimator, the latent speech representation attends to the text:

```
Sequence A (latent):  L [1, T_latent, 512]   ← speech being generated
Sequence B (text):    E [1, T_text, 256]      ← encoded text

Weight matrices (stored in vector_estimator.onnx):
  Wq [512, 512]  ← projects from LATENT dimension
  Wk [256, 512]  ← projects from TEXT dimension
  Wv [256, 512]  ← projects from TEXT dimension

Q = L × Wq  →  [1, T_latent, 512]     Queries come from LATENT (speech)
K = E × Wk  →  [1, T_text, 512]       Keys come from TEXT
V = E × Wv  →  [1, T_text, 512]       Values come from TEXT

Scores = Q × Kᵀ  →  [1, T_latent, T_text]

  ┌─────────────────────────────────────────────┐
  │           T_text (characters)               │
  │        H   e   l   l   o       w   o   r   │
  │  ┌─────────────────────────────────────┐    │
  │  │ .8  .1  .0  .0  .0  .0  .1  .0  .0 │ ← latent frame 0 (~0-70ms)    │
  │  │ .2  .6  .1  .0  .0  .0  .0  .0  .0 │ ← latent frame 1 (~70-140ms)  │
  │  │ .0  .1  .7  .1  .0  .0  .0  .0  .0 │ ← latent frame 2              │
  │  │ .0  .0  .1  .6  .2  .0  .0  .0  .0 │ ← latent frame 3              │
  │  │ ...                                  │                                │
  │  └─────────────────────────────────────┘    │
  │  ↑ Each row sums to 1.0 after softmax       │
  │  ↑ This IS the text-to-speech alignment     │
  └─────────────────────────────────────────────┘

Output = softmax(Scores) × V  →  [1, T_latent, 512]
```

**This is how the model knows which character to pronounce at which point in time.** Latent frame 0 attends mostly to "H", so it generates the /h/ sound. The model **learns this alignment automatically** during training.

### Multi-Head Attention

From tts.json: `"n_heads": 4`

```
512-dim ÷ 4 heads = 128-dim per head

Head 0: Q₀[128], K₀[128], V₀[128]  → might learn phonetic alignment
Head 1: Q₁[128], K₁[128], V₁[128]  → might learn stress patterns
Head 2: Q₂[128], K₂[128], V₂[128]  → might learn intonation cues
Head 3: Q₃[128], K₃[128], V₃[128]  → might learn punctuation effects

Each head computes its own [T_latent, T_text] attention matrix
Then concatenate → [T_latent, 512] → output projection
```

---

## 10. LARoPE — Length-Aware Rotary Position Embedding

### The Position Problem in Cross-Attention

Cross-attention has **no concept of position**. The score between latent frame 5 and text character 3 depends only on their *content*, not *where* they are. Without position information, the model has no built-in way to know that early latent frames should attend to early text characters.

### Standard RoPE (Rotary Position Embedding)

RoPE encodes position by **rotating** query and key vectors based on position index, applied in 2D pairs:

```
For a vector at position p, with dimension pairs (d₀,d₁), (d₂,d₃), ...

  Before RoPE:  [q₀, q₁, q₂, q₃, ...]

  After RoPE:   [q₀·cos(pθ₀) - q₁·sin(pθ₀),    ← pair (0,1) rotated
                  q₀·sin(pθ₀) + q₁·cos(pθ₀),
                  q₂·cos(pθ₁) - q₃·sin(pθ₁),    ← pair (2,3) rotated
                  q₂·sin(pθ₁) + q₃·cos(pθ₁),
                  ...]

  Where θᵢ = 1 / (base^(2i/dim))     base = 10000 (from tts.json)
```

Key property: dot product between positions p and q depends on **relative distance** (p-q). Nearby positions naturally score higher.

### The Problem with Standard RoPE for TTS

```
Text:    "Hello world."  →  maybe 20 tokens
Latent:  corresponding audio  →  maybe 50 frames

Standard RoPE positions:
  Text:    0, 1, 2, ..., 19
  Latent:  0, 1, 2, ..., 49

Problem: Position 19 (text end) and position 49 (latent end) are both "the end",
         but RoPE makes them look far apart (distance = 30).

         Position 10 (text middle) and position 25 (latent middle)
         have distance 15, even though they should align.
```

### LARoPE: The Fix

LARoPE **rescales** position indices so both sequences span the same effective range:

```
From tts.json: "rotary_base": 10000, "rotary_scale": 10

Text length:   T_text = 20
Latent length: T_latent = 50

Standard RoPE positions:
  Text:    [0, 1, 2, 3, ..., 19]
  Latent:  [0, 1, 2, 3, ..., 49]

LARoPE normalized positions:
  Text:    [0/20, 1/20, 2/20, ..., 19/20]  × scale  =  [0.0, 0.5, 1.0, ..., 9.5]
  Latent:  [0/50, 1/50, 2/50, ..., 49/50]  × scale  =  [0.0, 0.2, 0.4, ..., 9.8]

Now:
  Text end (9.5) and Latent end (9.8) → very close! ✓
  Text middle (4.75) and Latent middle (4.9) → very close! ✓
```

The `rotary_scale` controls how strongly the model biases toward diagonal alignment.

### Actual Computation

```
1. Compute Q from latent, K from text (standard projections)

2. Compute normalized positions:
   pos_q[i] = (i / T_latent) × 10     for each latent frame
   pos_k[j] = (j / T_text) × 10       for each text token

3. Compute rotation angles:
   θ_dim = 1 / (10000^(2d/dim_per_head))    for each dimension pair d
   angle_q[i,d] = pos_q[i] × θ_dim
   angle_k[j,d] = pos_k[j] × θ_dim

4. Apply rotations to Q and K (element-wise sin/cos multiply)

5. Compute attention: rotated_Q × rotated_Kᵀ → scores
```

The sin/cos values are **computed on-the-fly** from position indices and base/scale constants — not stored parameters. For QCS6490, verify that QNN supports element-wise sin/cos rotations, or pre-compute rotation matrices for fixed sequence lengths.

### Complete Cross-Attention + LARoPE in One Vector Estimator Block

```
Latent [1, 512, T_latent]
    │
    ▼
Time Conditioning                    ← adds timestep info
    │
    ▼
Style Conditioning                   ← adds voice style
    │
    ▼
Cross-Attention with LARoPE          ← THE KEY COMPONENT
    │   Q from latent: [T_latent, 128] × 4 heads
    │   K from text:   [T_text, 128] × 4 heads
    │   V from text:   [T_text, 128] × 4 heads
    │   LARoPE applied to Q and K before dot product
    │   Scores: [T_latent, T_text] × 4 heads
    │   Output: [T_latent, 512]
    │
    │   Stored weight matrices:
    │     Wq: [512, 512]    ← 262,144 floats
    │     Wk: [256, 512]    ← 131,072 floats
    │     Wv: [256, 512]    ← 131,072 floats
    │     Wo: [512, 512]    ← 262,144 floats (output projection)
    │
    ▼
ConvNeXt_0 (4 layers, dilations [1,2,4,8])  ← local temporal processing
    │
    ▼
ConvNeXt_1 (1 layer)
    │
    ▼
ConvNeXt_2 (1 layer)
    │
    ▼
Output [1, 512, T_latent]  → feeds into next block
```

The cross-attention handles **what** to say at each time position (alignment).
The ConvNeXt blocks handle **how** to shape local acoustic features (smoothing, details).
LARoPE ensures alignment naturally follows left-to-right reading order.

---

## 11. How Convolution Sliding Works

The kernel window **slides one position at a time** across every input:

```
Input:  [x₁, x₂, x₃, x₄, x₅, x₆, x₇, x₈, x₉, x₁₀]
         ↑_____________↑
         kernel window (5)


Position 1:  [x₁, x₂, x₃, x₄, x₅]  →  output₁ = x₁·w₁ + x₂·w₂ + x₃·w₃ + x₄·w₄ + x₅·w₅
              ↓ slide right by 1

Position 2:  [x₂, x₃, x₄, x₅, x₆]  →  output₂ = x₂·w₁ + x₃·w₂ + x₄·w₃ + x₅·w₄ + x₆·w₅
              ↓ slide right by 1

Position 3:  [x₃, x₄, x₅, x₆, x₇]  →  output₃ = x₃·w₁ + x₄·w₂ + x₅·w₃ + x₆·w₄ + x₇·w₅
              ↓ slide right by 1

Position 4:  [x₄, x₅, x₆, x₇, x₈]  →  output₄ = ...
Position 5:  [x₅, x₆, x₇, x₈, x₉]  →  output₅ = ...
Position 6:  [x₆, x₇, x₈, x₉, x₁₀] →  output₆ = ...
```

**The same 5 kernel weights are reused at every position.** For input of length T, output is length T (with padding).

Each input position influences **five different outputs** and gets "seen" from different window positions.

### Concrete Example — "Hello" in Text Encoder

```
"H"  "e"  "l"  "l"  "o"     ← 5 characters
 ↓    ↓    ↓    ↓    ↓
x₁   x₂   x₃   x₄   x₅    ← each is [256] after character embedding

With padding (kernel 5 needs 2 positions on each side):

 [0,  0,  x₁, x₂, x₃, x₄, x₅, 0,  0]

Position 1: pad, pad, H, e, l   → output for "H" — sees context of "He" and "l"
Position 2: pad, H, e, l, l     → output for "e" — sees "H", "e", "l", "l"
Position 3: H, e, l, l, o       → output for "l" — sees full word "Hello"
Position 4: e, l, l, o, pad     → output for "l" — sees "ello"
Position 5: l, l, o, pad, pad   → output for "o" — sees "llo"

Output: 5 positions → same length as input
```

### Stacking Layers Expands Receptive Field

When you stack 6 layers (like in the text encoder), the second layer's kernel-5 window looks at outputs that already incorporated 5 positions each:
- Layer 1: each position sees ~5 original positions
- Layer 2: sees ~9 positions
- Layer 3: sees ~13 positions
- ...
- Layer 6: sees ~25 positions — enough to cover most words

---

## 12. QCS6490 Conversion Considerations

### Model Sizes and Computational Profile

| Model | Size | Runs Per Utterance | Compute Profile |
|-------|------|-------------------|-----------------|
| duration_predictor | ~1.5 MB | 1× | Light — small MLP + convolutions |
| text_encoder | ~21-27 MB | 1× | Medium — attention + convolutions |
| vector_estimator | ~132-175 MB | N× (2-10 times) | **Heavy** — cross-attention + deep convolutions |
| vocoder | ~55-101 MB | 1× | Medium-Heavy — deep ConvNeXt decoder |

### Key Architecture Characteristics for QNN/SNPE Conversion

1. **ConvNeXt Blocks:** 1D depthwise + pointwise convolutions, GELU, LayerNorm, residual connections. Hexagon DSP should handle these well — among the best-optimized operations on HTP.

2. **Multi-Head Attention:** Present in text encoder (4 heads) and vector estimator cross-attention (4 heads). Vector estimator uses **Rotary Position Embeddings (RoPE)** with custom LARoPE scaling — verify RoPE op support in QNN.

3. **Flow Matching Loop:** Keep as application-level loop calling the model N times. Use 2 steps for maximum speed.

4. **Dynamic Shapes:** All models have dynamic sequence lengths. Options:
   - Use dynamic shape support in QNN
   - Pad to fixed sizes and mask
   - Create multiple model variants for different length buckets

5. **Operators to Verify:**
   - 1D Convolutions (with various dilations: 1,2,4,8)
   - Multi-head attention (with rotary embeddings)
   - GELU activation
   - Layer Normalization
   - Einsum operations (in style token attention)
   - Interpolation/upsampling (in vocoder)

6. **Recommended Conversion Path:**
   ```
   ONNX → onnx-simplifier → QNN/SNPE conversion tool
   ```
   Models are already OnnxSlim-optimized on HuggingFace.

7. **Quantization Notes:**
   - Vocoder is most sensitive to quantization — consider keeping in FP16
   - duration_predictor and text_encoder may tolerate INT8 well
   - vector_estimator: test FP16 first; cross-attention layers are precision-sensitive

### Memory Footprint Estimate

- Total ONNX model size: ~260-305 MB
- Runtime activations: typically < 50 MB (depends on sequence length)
- Voice style vectors: < 1 MB per voice
- Total RAM needed: ~350-400 MB estimated (well within 16GB LPDDR5)

---

## Architecture Diagram (from tts.json)

```
                        ┌──────────────────────────────────────────┐
                        │           SPEECH AUTOENCODER (ae)        │
                        │                                          │
  [Audio 44.1kHz] ──►   │  Encoder: Mel Spec → 10×ConvNeXt → 24d  │  ◄── Training only
                        │  Decoder: 24d → 10×ConvNeXt → Audio     │  ◄── vocoder.onnx
                        └──────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────┐
  │              TEXT-TO-LATENT MODULE (ttl)                        │
  │                                                                 │
  │  ┌─────────────┐   ┌──────────────┐   ┌──────────────────┐    │
  │  │ Text Encoder │   │Style Encoder │   │ Speech-Prompted  │    │
  │  │ char_emb 256 │   │ 50 tokens    │   │ Text Encoder     │    │
  │  │ 6×ConvNeXt   │──►│ 2-head attn  │──►│ cross-attention  │    │
  │  │ 4×Attn       │   │ 256-dim      │   │ text+style→256d  │    │
  │  └──────────────┘   └──────────────┘   └────────┬─────────┘    │
  │                                                  │              │
  │          ┌─── text_encoder.onnx ─────────────────┘              │
  │          │                                                      │
  │          ▼                                                      │
  │  ┌───────────────────────────────────────────────┐             │
  │  │           VECTOR FIELD (Flow Matching)         │             │
  │  │  proj_in: 144→512                              │             │
  │  │  4× Main Blocks:                               │             │
  │  │    ├─ Time conditioning (64d)                   │             │
  │  │    ├─ Style conditioning (256d)                 │             │
  │  │    ├─ Text cross-attention (4 heads, LARoPE)   │             │
  │  │    ├─ ConvNeXt [1,2,4,8] dilations             │             │
  │  │    └─ ConvNeXt ×2 (dilation 1)                 │             │
  │  │  Final ConvNeXt (4 layers)                     │             │
  │  │  proj_out: 512→144                             │             │
  │  └───────────────────────────────────────────────┘             │
  │          │                                                      │
  │          └─── vector_estimator.onnx                             │
  └─────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────┐
  │              DURATION PREDICTOR (dp)                            │
  │                                                                 │
  │  Sentence Encoder (64d, 6×ConvNeXt, 2×Attn)                   │
  │  Style Encoder (8 tokens, 16d values)                          │
  │  Predictor MLP (64+8×16 → 128 → 128 → 1)                     │
  │                                                                 │
  │  └─── duration_predictor.onnx                                  │
  └─────────────────────────────────────────────────────────────────┘
```

## Voice Style Format

```json
{
  "style_ttl": [[...50 rows × 256 cols...]],   // Shape: [50, 256]
  "style_dp":  [[...8 rows × 16 cols...]]       // Shape: [8, 16]
}
```

Pre-extracted from reference audio. At inference, loaded and fed directly — no audio encoding needed.

---

## Summary Table

| Component | ONNX Model | tts.json Section | Key Dimensions | Purpose |
|-----------|-----------|------------------|----------------|---------|
| Text Encoder | `text_encoder.onnx` | `ttl.text_encoder`, `ttl.style_encoder`, `ttl.speech_prompted_text_encoder` | In: chars, Out: [seq, 256] | Encode text + voice style |
| Duration Predictor | `duration_predictor.onnx` | `dp` | In: chars, Out: scalar (samples) | Predict speech length |
| Vector Estimator | `vector_estimator.onnx` | `ttl.vector_field`, `ttl.flow_matching` | In/Out: [1, 144, T] | Flow-match noise → latent speech |
| Vocoder | `vocoder.onnx` | `ae.decoder` | In: [1, 24, T×6], Out: waveform | Decode latent → audio |
| Unicode Indexer | N/A (JSON lookup) | `ttl.text_encoder.char_dict_path` | ~262 KB map | Char → integer index |
| TTS Config | N/A (JSON) | Entire file | 8.65 KB | All architecture hyperparameters |

---

*Document generated from architecture learning session — targeting Qualcomm QCS6490 deployment with QAIRT 2.37 and QNN SDK.*