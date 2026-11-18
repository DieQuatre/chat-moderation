"""Transformer models (area C + D): fine-tuned BERT in FP32 (PyTorch) and INT8 (ONNX Runtime).

Both read the same normalized text, cut to 128 WordPiece tokens.
"""
from pathlib import Path

import numpy as np

from src.text.normalize import normalize

ROOT = Path(__file__).resolve().parents[2]
BASE_MODEL = "bert-base-uncased"
MAX_LEN = 128
FP32_DIR = ROOT / "models" / "bert-fp32"
ONNX_DIR = ROOT / "models" / "bert-onnx"
ONNX_FP32 = ONNX_DIR / "model_fp32.onnx"
ONNX_INT8 = ONNX_DIR / "model_int8.onnx"


def softmax(logits):
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def encode(tokenizer, texts, kind):
    return tokenizer([normalize(t) for t in texts], truncation=True, max_length=MAX_LEN,
                     padding=True, return_tensors=kind)


class BertFP32:
    name = "BERT FP32"

    def __init__(self, path=FP32_DIR, device="cpu"):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path).to(device).eval()

    def predict_proba(self, texts, batch_size=32):
        texts, out = list(texts), []
        with self.torch.no_grad():
            for i in range(0, len(texts), batch_size):
                enc = encode(self.tokenizer, texts[i:i + batch_size], "pt").to(self.device)
                logits = self.model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits
                out.append(softmax(logits.float().cpu().numpy())[:, 1])
        return np.concatenate(out) if out else np.zeros(0)


class BertInt8:
    name = "BERT INT8"

    def __init__(self, path=ONNX_INT8):
        import onnxruntime as ort
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(FP32_DIR)
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])

    def predict_proba(self, texts, batch_size=32):
        texts, out = list(texts), []
        for i in range(0, len(texts), batch_size):
            enc = encode(self.tokenizer, texts[i:i + batch_size], "np")
            feed = {"input_ids": enc["input_ids"].astype(np.int64),
                    "attention_mask": enc["attention_mask"].astype(np.int64)}
            out.append(softmax(self.session.run(None, feed)[0])[:, 1])
        return np.concatenate(out) if out else np.zeros(0)
