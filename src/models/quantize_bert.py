"""Quantization (area D): FP32 BERT -> ONNX -> dynamic INT8 with ONNX Runtime.

Same method as the lecture "Making Transformers Efficient in Production":
export the graph to ONNX, then onnxruntime.quantization.quantize_dynamic(QInt8).

Run:  python -m src.models.quantize_bert   (needs models/bert-fp32/, writes models/bert-onnx/)
"""
import numpy as np
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic

from src.data.load import load_split
from src.eval.metrics import report
from src.models.bert import FP32_DIR, ONNX_DIR, ONNX_FP32, ONNX_INT8, BertFP32, BertInt8, encode


class LogitsOnly(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits


def export_onnx():
    fp32 = BertFP32(FP32_DIR, device="cpu")
    sample = encode(fp32.tokenizer, ["you are a nice person", "hello"], "pt")
    args = (sample["input_ids"], sample["attention_mask"])
    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    try:  # classic exporter with dynamic batch and sequence length
        torch.onnx.export(LogitsOnly(fp32.model), args, str(ONNX_FP32), dynamo=False,
                          input_names=["input_ids", "attention_mask"], output_names=["logits"],
                          dynamic_axes={"input_ids": {0: "batch", 1: "seq"},
                                        "attention_mask": {0: "batch", 1: "seq"},
                                        "logits": {0: "batch"}},
                          opset_version=17)
    except Exception as e:  # newer PyTorch: dynamo exporter
        print("Classic exporter not available, using dynamo exporter:", type(e).__name__)
        batch, seq = torch.export.Dim("batch"), torch.export.Dim("seq", max=512)
        program = torch.onnx.export(LogitsOnly(fp32.model), args, dynamo=True,
                                    input_names=["input_ids", "attention_mask"], output_names=["logits"],
                                    dynamic_shapes={"input_ids": {0: batch, 1: seq},
                                                    "attention_mask": {0: batch, 1: seq}})
        program.save(str(ONNX_FP32), external_data=False)
    return fp32


def main():
    fp32 = export_onnx()
    print("Exported", ONNX_FP32)
    quantize_dynamic(str(ONNX_FP32), str(ONNX_INT8), weight_type=QuantType.QInt8)
    print("Quantized", ONNX_INT8)

    for path in (ONNX_FP32, ONNX_INT8):
        print(f"{path.name}: {path.stat().st_size / 2**20:.1f} MB")

    # sanity check: the ONNX models must agree with PyTorch
    checks = ["thanks for fixing the article", "you are a stupid idiot", "ok"]
    p_torch = fp32.predict_proba(checks)
    p_onnx32 = BertInt8(ONNX_FP32).predict_proba(checks)
    int8 = BertInt8(ONNX_INT8)
    p_int8 = int8.predict_proba(checks)
    print("PyTorch FP32:", np.round(p_torch, 4))
    print("ONNX FP32:   ", np.round(p_onnx32, 4))
    print("ONNX INT8:   ", np.round(p_int8, 4))

    val_texts, val_labels = load_split("val")
    report(BertInt8.name, val_labels, int8.predict_proba(val_texts))


if __name__ == "__main__":
    main()
