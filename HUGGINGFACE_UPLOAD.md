# Files to Upload to HuggingFace

This document lists all large files that should be uploaded to HuggingFace model repository.

## Directory Structure for HuggingFace

```
supertonic2-qcs6490/
├── README.md (copy from main repo)
├── QNN_Model_lib/
│   └── aarch64-oe-linux-gcc11.2/
│       ├── libduration_predictor_htp.so (1.0 MB)
│       ├── libtext_encoder_htp.so (7.5 MB)
│       ├── libvector_estimator_htp.so (34 MB)
│       └── libvocoder_htp.so (25 MB)
│
├── QNN_Models/
│   ├── duration_predictor_htp.cpp (1.5 MB)
│   ├── duration_predictor_htp.bin (460 KB)
│   ├── duration_predictor_htp_net.json (722 KB)
│   ├── text_encoder_htp.cpp (2.1 MB)
│   ├── text_encoder_htp.bin (6.7 MB)
│   ├── text_encoder_htp_net.json (1.1 MB)
│   ├── vector_estimator_htp.cpp (4.7 MB)
│   ├── vector_estimator_htp.bin (32 MB)
│   ├── vector_estimator_htp_net.json (2.3 MB)
│   ├── vocoder_htp.cpp (1.2 MB)
│   ├── vocoder_htp.bin (25 MB)
│   └── vocoder_htp_net.json (581 KB)
│
├── model/ (ONNX models and configs)
├── calibration_data/ (10 samples)
└── samples/ (example outputs)
```

## Total Size: ~400 MB

## Upload Using HuggingFace CLI

```bash
pip install huggingface_hub
huggingface-cli login
huggingface-cli repo create supertonic2-qcs6490 --type model

# Upload directories
huggingface-cli upload <username>/supertonic2-qcs6490 QNN_Model_lib
huggingface-cli upload <username>/supertonic2-qcs6490 QNN_Models
huggingface-cli upload <username>/supertonic2-qcs6490 model
huggingface-cli upload <username>/supertonic2-qcs6490 calibration_data
huggingface-cli upload <username>/supertonic2-qcs6490 board_outputs/samples
```
