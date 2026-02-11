# From Text to Talk: Audio-Language Model Needs Non-Autoregressive Joint Training

This repository contains the official implementation of our paper "From Text to Talk: Audio-Language Model Needs Non-Autoregressive Joint Training".

## Project Structure

```
TtT/
├── README.md                         # This file
├── train_TtT.py                      # Main training script
├── inference_TtT.py                  # Inference script
├── modeling_qwen_TtT.py              # Custom Qwen2 model with AR+Diffusion
├── ds_zero_3.json                    # DeepSpeed Zero-3 configuration
├── data_config/
│   └── data_config_TtT.json          # Data configuration file
├── datasets/                         # Training datasets
│   ├── asr_tts_samples.json                  # ASR/TTS data
│   ├── audio_chat_samples.json               # Audio chat data
│   ├── interleaved_data_en_samples.json      # English interleaved data
│   ├── interleaved_data_zh_samples.json      # Chinese interleaved data
│   ├── sec_aac_asc_task_samples.json         # Speech classification tasks
│   └── text_chat_samples.json                # Text chat data
└── exp_shells/
    └── train_TtT.sh                  # Training launch script
```

## Installation

### Requirements

- Python 3.11+
- PyTorch 2.7+
- Transformers
- DeepSpeed
- PEFT
- Accelerate

### Setup Environment

```bash
# Clone the repository
git clone <repository-url>
cd TtT

# Install dependencies (you may need to create requirements.txt)
pip install torch transformers deepspeed peft accelerate wandb
pip install bitsandbytes  # For quantization support
```

## Data Preparation

### Dataset Format

The model supports two main data formats:

1. **Conversation Format** (`messages` field):
```json
{
  "messages": [
    {"role": "system", "content": "System prompt"},
    {"role": "user", "content": "User input with <|audio_*|> tokens"},
    {"role": "assistant", "content": "Response with mixed text and audio tokens"}
  ]
}
```

2. **Plain Text Format** (`text` field):
```json
{
  "text": "Direct text content for training"
}
```

### Audio Token Format

Audio content is represented using special tokens:
- `<|begin_of_audio|>` / `<|begin_of_quad|>`: Start of audio segment
- `<|audio_XXXXX|>`: Audio tokens with specific IDs
- `<|end_of_audio_new|>` / `<|end_of_audio_span_last|>`: End of audio segment


## Training

### Configuration

1. **Model Configuration**: Update paths in `exp_shells/train_TtT.sh`
   - `MODEL_PATH`: Base model path (e.g., Qwen2-7B)
   - `DATA_PATH`: Path to data configuration file
   - `OUTPUT_DIR`: Training output directory

2. **Data Configuration**: Edit `data_config/data_config_TtT.json` to specify:
   - Dataset paths
   - Sample ratios
   - Data types (IT for Instruction Tuning, PT for Pre-Training)

### Training Command

```bash
# Single GPU training
python train_TtT.py \
    --model_name_or_path /path/to/qwen2-model \
    --data_path data_config/data_config_TtT.json \
    --output_dir ./output \
    --num_train_epochs 10 \
    --bf16 True \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --model_max_length 2048 \
    --learning_rate 2e-5 \
    --use_lora False

# Multi-GPU training with DeepSpeed
bash exp_shells/train_TtT.sh
```

### Key Training Parameters

- `--unmasked_audio_prob`: Probability of training on clean audio (default: 0.3)
- `--prefix_preservation_ratio`: Ratio of preserving audio prefix during diffusion (default: 0.3)
- `--quad_span_truncation_prob`: Probability of truncating quad spans (default: 0.5)
- `--model_max_length`: Maximum sequence length (default: 2048)

## Inference

### Basic Inference

```python
from modeling_qwen_TtT import Qwen2ForARDiffLM
from transformers import AutoTokenizer

# Load model and tokenizer
model = Qwen2ForARDiffLM.from_pretrained("/path/to/checkpoint")
tokenizer = AutoTokenizer.from_pretrained("/path/to/checkpoint")

# Set up special tokens
if tokenizer.mask_token_id is None:
    tokenizer.mask_token_id = tokenizer.convert_tokens_to_ids("<|mask_token|>")

# Generate
prompt = "<|im_start|>user\n<|begin_of_audio|><|audio_1234|>...<|end_of_audio|><|im_end|>\n<|im_start|>assistant\n"
output = generate(model, tokenizer, prompt, max_gen_len=2048)
```

### Advanced Generation

```python
# Run inference script
python inference_TtT.py
```

### Generation Parameters

- `max_gen_len`: Maximum generation length
- `diffusion_steps`: Number of diffusion steps per block
- `diffusion_gen_length`: Maximum diffusion generation length
- `block_length`: Block length for diffusion generation
- `ar_temperature` / `diffusion_temperature`: Sampling temperatures
- `cfg_scale`: Classifier-free guidance scale
- `top_k` / `top_p`: Nucleus sampling parameters

## Configuration

### DeepSpeed Configuration

The project uses DeepSpeed Zero-3 for efficient training:

```json
{
  "train_micro_batch_size_per_gpu": "auto",
  "zero_optimization": {
    "stage": 3,
    "allgather_partitions": true,
    "overlap_comm": true,
    "stage3_gather_16bit_weights_on_model_save": true
  }
}
```

### Data Configuration

Configure datasets in `data_config/data_config_TtT.json`:

```json
[
  {
    "path": "/path/to/dataset.json",
    "sample_ratio": 1.0,
    "data_type": "IT"  // or "PT"
  }
]
```
