"""Neural network (area A + B): our own word2vec + a feedforward network trained with SGD.

1. word2vec (skip-gram, 100 dims) trained on the training comments only.
2. A comment = the average of its word vectors.
3. Feedforward network 100 -> 64 (ReLU) -> 2, trained with SGD + backpropagation,
   class-weighted cross-entropy (classes are imbalanced). The epoch with the best
   val macro-F1 is kept.

Run:  python -m src.models.w2v_ffn      (saves models/w2v.model and models/ffn.pt)
"""
from pathlib import Path

import numpy as np
import torch
from gensim.models import Word2Vec
from torch import nn

from src.text.normalize import tokens

ROOT = Path(__file__).resolve().parents[2]
W2V_FILE = ROOT / "models" / "w2v.model"
FFN_FILE = ROOT / "models" / "ffn.pt"
DIM, HIDDEN, EPOCHS, BATCH, LR = 100, 64, 30, 64, 0.05
SEED = 42


def build_net():
    return nn.Sequential(nn.Linear(DIM, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, 2))


class W2VFFN:
    name = "word2vec + FFN"

    def __init__(self, vectors, net):
        self.vectors = vectors  # gensim KeyedVectors
        self.net = net.eval()

    def embed(self, texts):
        out = np.zeros((len(texts), DIM), dtype=np.float32)
        for i, text in enumerate(texts):
            known = [self.vectors[w] for w in tokens(text) if w in self.vectors]
            if known:
                out[i] = np.mean(known, axis=0)
        return out

    def predict_proba(self, texts):
        with torch.no_grad():
            logits = self.net(torch.from_numpy(self.embed(list(texts))))
            return torch.softmax(logits, dim=1)[:, 1].numpy()

    @classmethod
    def load(cls):
        net = build_net()
        net.load_state_dict(torch.load(FFN_FILE))
        return cls(Word2Vec.load(str(W2V_FILE)).wv, net)

    @classmethod
    def train(cls, train_texts, train_labels, val_texts, val_labels):
        from src.eval.metrics import macro_f1

        torch.manual_seed(SEED)
        print("Training word2vec ...")
        w2v = Word2Vec([tokens(t) for t in train_texts], vector_size=DIM, window=5,
                       min_count=2, sg=1, epochs=5, workers=4, seed=SEED)
        print(f"Vocabulary: {len(w2v.wv):,} words")

        model = cls(w2v.wv, build_net())
        x_train = torch.from_numpy(model.embed(train_texts))
        y_train = torch.from_numpy(np.asarray(train_labels)).long()
        x_val = val_texts

        counts = np.bincount(train_labels, minlength=2)
        weights = torch.tensor(len(train_labels) / (2 * counts), dtype=torch.float32)
        loss_fn = nn.CrossEntropyLoss(weight=weights)
        optimizer = torch.optim.SGD(model.net.parameters(), lr=LR, momentum=0.9)

        best_f1, best_state = -1.0, None
        x_val_vec = torch.from_numpy(model.embed(x_val))
        for epoch in range(1, EPOCHS + 1):
            model.net.train()
            order = torch.randperm(len(y_train))
            for start in range(0, len(order), BATCH):
                idx = order[start:start + BATCH]
                optimizer.zero_grad()
                loss = loss_fn(model.net(x_train[idx]), y_train[idx])
                loss.backward()      # backpropagation
                optimizer.step()     # SGD update
            model.net.eval()
            with torch.no_grad():
                probs = torch.softmax(model.net(x_val_vec), dim=1)[:, 1].numpy()
            f1 = macro_f1(val_labels, probs)
            print(f"epoch {epoch:2d}  loss {loss.item():.4f}  val macro-F1 {f1:.4f}")
            if f1 > best_f1:
                best_f1 = f1
                best_state = {k: v.clone() for k, v in model.net.state_dict().items()}

        model.net.load_state_dict(best_state)
        model.net.eval()
        W2V_FILE.parent.mkdir(exist_ok=True)
        w2v.save(str(W2V_FILE))
        torch.save(best_state, FFN_FILE)
        return model


if __name__ == "__main__":
    from src.data.load import load_split
    from src.eval.metrics import report

    train_texts, train_labels = load_split("train")
    val_texts, val_labels = load_split("val")
    model = W2VFFN.train(train_texts, train_labels, val_texts, val_labels)
    report(model.name, val_labels, model.predict_proba(val_texts))
    print("Saved", W2V_FILE, "and", FFN_FILE)
