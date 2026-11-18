"""Upload the trained models to the Hugging Face Hub (run once, after `hf auth login`).

Only the files the app needs are uploaded (about 560 MB); the FP32 ONNX file is skipped.
Run:  python -m src.models.upload_models
"""
from huggingface_hub import create_repo, upload_folder

from src.models.download import HF_REPO, ROOT

FILES = ["known_words.txt", "tfidf_lr.joblib", "w2v.model", "w2v.model.*", "ffn.pt",
         "bert-fp32/*", "bert-onnx/model_int8.onnx"]

if __name__ == "__main__":
    create_repo(HF_REPO, repo_type="model", exist_ok=True)
    upload_folder(repo_id=HF_REPO, folder_path=ROOT / "models", allow_patterns=FILES,
                  commit_message="Trained chat moderation models")
    print(f"Uploaded: https://huggingface.co/{HF_REPO}")
