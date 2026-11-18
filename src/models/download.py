"""Download the trained models from the Hugging Face Hub if they are not on disk yet."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HF_REPO = "kubraalkan/chat-moderation-models"
NEEDED = ["models/known_words.txt", "models/tfidf_lr.joblib", "models/w2v.model", "models/ffn.pt",
          "models/bert-fp32/model.safetensors", "models/bert-onnx/model_int8.onnx"]


def ensure_models():
    if all((ROOT / f).exists() for f in NEEDED):
        return
    from huggingface_hub import snapshot_download

    print(f"Downloading models from huggingface.co/{HF_REPO} (about 560 MB, only once) ...")
    snapshot_download(HF_REPO, local_dir=ROOT / "models")
