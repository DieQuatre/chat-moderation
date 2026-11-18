"""Fine-tune bert-base-uncased for clean / offensive classification (area C).

Plain PyTorch loop: AdamW, lr 2e-5 with linear warmup + decay, batch 8, 2 epochs,
fp16 mixed precision on the GPU. The epoch with the best val macro-F1 is saved.

Run:  python -m src.models.train_bert      (saves models/bert-fp32/)
"""
import time

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.data.load import load_split
from src.eval.metrics import macro_f1, report
from src.models.bert import BASE_MODEL, FP32_DIR, BertFP32, encode

BATCH, EPOCHS, LR, WARMUP, SEED = 8, 2, 2e-5, 0.06, 42  # batch 8 fits in 4 GB VRAM


def main():
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    train_texts, train_labels = load_split("train")
    val_texts, val_labels = load_split("val")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=2).to(device)

    steps = EPOCHS * ((len(train_texts) + BATCH - 1) // BATCH)
    warmup = int(WARMUP * steps)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: s / max(1, warmup) if s < warmup else max(0.0, (steps - s) / max(1, steps - warmup)))
    scaler = torch.amp.GradScaler(device, enabled=device == "cuda")
    labels_all = torch.tensor(train_labels, dtype=torch.long)

    print("Training starts", flush=True)
    best_f1, step, start = -1.0, 0, time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = torch.randperm(len(train_texts)).tolist()
        for i in range(0, len(order), BATCH):
            idx = order[i:i + BATCH]
            enc = encode(tokenizer, [train_texts[j] for j in idx], "pt").to(device)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
                out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                            labels=labels_all[idx].to(device))
            optimizer.zero_grad()
            scaler.scale(out.loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            step += 1
            if step <= 20 and device == "cuda":  # first steps: show speed and GPU memory
                torch.cuda.synchronize()
                print(f"step {step:2d}  {time.time() - start:6.1f} s  "
                      f"GPU memory {torch.cuda.memory_reserved() / 2**30:.2f} GB", flush=True)
            if step % 100 == 0:
                print(f"epoch {epoch} step {step}/{steps}  loss {out.loss.item():.4f}  "
                      f"{(time.time() - start) / 60:.1f} min", flush=True)

        model.eval()
        clf = BertFP32.__new__(BertFP32)
        clf.torch, clf.device, clf.tokenizer, clf.model = torch, device, tokenizer, model
        f1 = macro_f1(val_labels, clf.predict_proba(val_texts, batch_size=64))
        print(f"== epoch {epoch}: val macro-F1 {f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            model.save_pretrained(FP32_DIR)
            tokenizer.save_pretrained(FP32_DIR)
            print("   saved", FP32_DIR)

    best = BertFP32(FP32_DIR, device=device)
    report(BertFP32.name, val_labels, best.predict_proba(val_texts, batch_size=64))
    print(f"Done in {(time.time() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()
