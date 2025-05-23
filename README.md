## Models for IBM i
This repository provides an easy-to-follow process for downloading models from Hugging Face and transferring them to IBM i using `scp` or another transfer method. The models can then be used on IBM i with Llama.cpp.

## Requirements
- uv python package manager

## Usage

1. Install the `uv` package manager if you haven't already:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Clone the repo
    ```bash
    git clone https://github.com/ajshedivy/models-for-i.git
    cd models-for-i
    ```
3. Create a Hugging Face account and create an access token
   - Go to [Hugging Face](https://huggingface.co/) and create an account if you don't have one.
   - Go to your account settings and create a new access token. 
  
4. Login into Hugging Face using the command line:
   ```bash
   uv run huggingface-cli login
   ```
   Enter your access token when prompted.

5. Download the model

    ```bash
    uv run download_model.py models-for-i/Llama-3.2-1B-Instruct
    ```
    This will download the model to the `models/Llama-3.2-1B-Instruct` directory.

6. Transfer the model to IBM i using `scp` or your preferred transfer method. For example:
   ```bash
   ssh USER@HOST 'mkdir -p ~/models'
   scp models/Llama-3.2-1B-Instruct/Llama-3.2-1B-Instruct-f32-be.gguf USER@HOST:~/models/Llama-3.2-1B-Instruct-f32-be.gguf
   ```


