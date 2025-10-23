# Models for IBM i

This repository provides tools and workflows for downloading LLM models from Hugging Face and preparing them for use on IBM i with Llama.cpp. Pre-converted Big Endian models are available for direct download, or you can convert any GGUF model yourself.

Browse available pre-converted models on the [Hugging Face Models for i page](https://huggingface.co/models-for-i).

## Table of Contents
- [Quick Start](#quick-start)
- [Prerequisites](#prerequisites)
- [Workflow Options](#workflow-options)
  - [Option 1: Use Pre-Converted Models](#option-1-use-pre-converted-models)
  - [Option 2: Convert Models Yourself](#option-2-convert-models-yourself)
- [Supported Models](#supported-models)
- [Tool Reference](#tool-reference)

## Quick Start

Download a pre-converted model directly on IBM i:

```bash
curl -L -o Llama-3.2-1B-Instruct-f32-be.gguf \
  https://huggingface.co/models-for-i/Llama-3.2-1B-Instruct/resolve/main/Llama-3.2-1B-Instruct-f32-be.gguf
```

The model is now ready to use with Llama.cpp on IBM i.

## Prerequisites

- **For downloading pre-converted models**: None (use `curl` or `wget` directly on IBM i)
- **For converting models locally**:
  - `uv` Python package manager
  - Hugging Face account and access token
  - `scp` or another file transfer method to IBM i

Install `uv`:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Workflow Options

### Option 1: Use Pre-Converted Models

Pre-converted Big Endian models are available on Hugging Face and can be downloaded directly to IBM i.

**Step 1: Download the model on IBM i**

```bash
curl -L -o Llama-3.2-1B-Instruct-f32-be.gguf \
  https://huggingface.co/models-for-i/Llama-3.2-1B-Instruct/resolve/main/Llama-3.2-1B-Instruct-f32-be.gguf
```

**Step 2: Use with Llama.cpp**

The model is ready to use immediately.

### Option 2: Convert Models Yourself

Convert any GGUF model from Hugging Face to Big Endian format for use on IBM i.

> **Note**: These steps must be performed on your **local machine** (not on IBM i). After conversion, you'll transfer the model to IBM i.

**Step 1: Set up your environment**

On your local machine, clone this repository:
```bash
git clone https://github.com/ajshedivy/models-for-i.git
cd models-for-i
```

**Step 2: Authenticate with Hugging Face**

Create a Hugging Face account and access token:
- Create an account at [Hugging Face](https://huggingface.co/)
- Go to account settings and create a new access token

Login via CLI:
```bash
uv run huggingface-cli login
```

**Step 3: Download a model**

Using the `hf` CLI (recommended):
```bash
hf download unsloth/gpt-oss-20b-GGUF gpt-oss-20b-Q4_0.gguf --local-dir models/gpt-oss-20b
```

**Step 4: Convert to Big Endian format**

```bash
uv run gguf_convert_endian.py models/gpt-oss-20b/gpt-oss-20b-Q4_0.gguf big
```

> **Note**: This modifies the file in-place. The original file will be converted to Big Endian format.

**Step 5: Transfer to IBM i**

```bash
ssh USER@HOST 'mkdir -p ~/models'
scp models/gpt-oss-20b/gpt-oss-20b-Q4_0-be.gguf USER@HOST:~/models/
```

## Supported Models

### Llama-3.2-1B-Instruct
A 1 billion parameter model fine-tuned for instruction following.

- **Format**: GGUF (f32, Big Endian)
- **Repository**: [models-for-i/Llama-3.2-1B-Instruct](https://huggingface.co/models-for-i/Llama-3.2-1B-Instruct)
- **Direct Download**:
  ```bash
  curl -L -o Llama-3.2-1B-Instruct-f32-be.gguf \
    https://huggingface.co/models-for-i/Llama-3.2-1B-Instruct/resolve/main/Llama-3.2-1B-Instruct-f32-be.gguf
  ```

## Tool Reference

### gguf_convert_endian.py

Convert GGUF model files between different byte orders.

**Usage**:
```bash
uv run gguf_convert_endian.py [-h] [--dry-run] [--verbose] model {big,little,native}
```

**Arguments**:
- `model`: GGUF format model filename
- `{big,little,native}`: Requested byte order

**Options**:
- `-h, --help`: Show help message
- `--dry-run`: Preview changes without modifying files
- `--verbose`: Increase output verbosity

**Example**:
```bash
uv run gguf_convert_endian.py models/model.gguf big --verbose
```




