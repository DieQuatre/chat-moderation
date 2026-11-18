"""Paket 0: checks that the environment is ready (GPU, libraries, Kaggle access)."""
import os
import platform
import sys
import time
from pathlib import Path


def section(title):
    print(f"\n== {title} ==")


section("Python")
print(sys.version)
print(platform.platform())

section("PyTorch")
import torch

print("torch", torch.__version__, "| CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print("GPU:", props.name, "| VRAM GB:", round(props.total_memory / 1e9, 2))
    a = torch.randn(2048, 2048, device="cuda")
    start = time.perf_counter()
    (a @ a).sum().item()
    print(f"GPU matmul OK ({(time.perf_counter() - start) * 1000:.1f} ms)")

section("Libraries")
for name in ["transformers", "accelerate", "sklearn", "pandas", "gensim", "onnx",
             "onnxscript", "onnxruntime", "gradio", "kagglehub", "huggingface_hub"]:
    try:
        module = __import__(name)
        print(f"{name:16} {getattr(module, '__version__', '?')}")
    except Exception as e:
        print(f"{name:16} FAILED: {e!r}")

try:
    import onnxruntime as ort
    print("onnxruntime providers:", ort.get_available_providers())
except Exception:
    pass

section("Kaggle")
token_file = Path.home() / ".kaggle" / "access_token"
print("token file found:", token_file.exists())
if "KAGGLE_API_TOKEN" not in os.environ and token_file.exists():
    os.environ["KAGGLE_API_TOKEN"] = token_file.read_text().strip()
try:
    import kagglehub

    path = kagglehub.competition_download("jigsaw-toxic-comment-classification-challenge")
    print("download OK:", path)
    print(sorted(os.listdir(path)))
except Exception as e:
    print("Kaggle FAILED:", repr(e))
