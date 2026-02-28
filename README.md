# From Text to Talk (TtT): Audio-Language Model Needs Non-Autoregressive Joint Training

<p align="center">
  📑 <a href="https://openreview.net/pdf/b508e8a8f6d4e8363700fb26a76921badee24671.pdf"><b>Paper (ICLR 2026)</b></a> &nbsp; | &nbsp;
  📖 <a href="https://arxiv.org/abs/2509.20072"><b>arXiv/PDF</b></a> &nbsp; | &nbsp;
  🤗 <a href="https://huggingface.co/Stephen-Lee/TtT-3B"><b>Models</b></a> &nbsp; | &nbsp;
  💜 <a href="https://demopage200.github.io/demo_TtT"><b>Demo</b></a> &nbsp; | &nbsp;
  🖥️ <a href="#quickstart"><b>Quickstart</b></a> &nbsp; | &nbsp;
  ©️ <a href="#citation"><b>Citation</b></a>
</p>

This repository provides the **official implementation** of **TtT (Text-to-Talk)**, an audio-language model that supports **speech-to-speech interaction** via **non-autoregressive joint training**.

## Highlights
- **Unified training** for audio-text interaction with **AR + non-autoregressive** components.
- Supports multiple data formats: **conversation-style `messages`** and **plain `text`**.
- DeepSpeed ZeRO-3 configuration included for scalable training.


## Project Structure
```
TtT/
├── README.md                         # This file
├── requirements.txt                  # Pinned dependencies for reproducible setup
├── train_TtT.py                      # Main training script
├── inference_TtT.py                  # Inference script
├── modeling_qwen_TtT.py              # Custom Qwen2 model with AR + Diffusion
├── ds_zero_3.json                    # DeepSpeed ZeRO-3 configuration
├── data_config/
│   └── data_config_TtT.json          # Data configuration file
├── datasets/                         # Training datasets (examples / templates)
│   ├── asr_tts_samples.json          # ASR/TTS data
│   ├── audio_chat_samples.json       # Audio chat data
│   ├── interleaved_data_en_samples.json  # English interleaved data
