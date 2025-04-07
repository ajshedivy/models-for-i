import argparse
from huggingface_hub import snapshot_download

def parse_model_name(model_id):
    return model_id.split('/')[-1]

def main():
    parser = argparse.ArgumentParser(description="Download model from Hugging Face Hub")
    parser.add_argument('model_id', type=str, help='The model ID from Hugging Face Hub')
    args = parser.parse_args()

    model_id = args.model_id
    local_dir = parse_model_name(model_id)
    
    snapshot_download(repo_id=model_id, local_dir=f'models/{local_dir}', local_dir_use_symlinks=False, revision="main")

if __name__ == "__main__":
    main()