# Real-Time Chat Moderation

SWE015 – Introduction to Large Language Models · Homework · **Use Case 4: Real-Time Chat Moderation**

A chat room where every message is labeled **clean / offensive** by a model before it is shown.
Offensive messages are masked for everyone, and a side panel logs each message's response time.
The app runs a **quantized (INT8) BERT** on the CPU and lets you switch to the other four models.

**Demo video (≤2 min):** https://youtu.be/SA1vTwfuYZE

## Quick start

Requires Python 3.13 (3.10+ should work). From the project folder:

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows   (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:7860. On the first run the trained models (~580 MB) are downloaded from
[huggingface.co/kubraalkan/chat-moderation-models](https://huggingface.co/kubraalkan/chat-moderation-models).
To chat as two players, open the page in a second (private) browser window, or run
`python app.py --share` for a temporary public link.

## Results (same held-out test set for every model)

10,000 comments from the official labeled Jigsaw test set, never used for training or tuning.

| Model | Test macro-F1 | Size (MB) | Latency (ms, mean ± std) |
|---|---|---|---|
| Lexicon rule (bad-word list + edit distance) | 0.7294 | 0.2 | 0.15 ± 0.20 |
| tf-idf + logistic regression | 0.7795 | 4.6 | 1.12 ± 0.43 |
| word2vec (own) + feedforward network | 0.7685 | 30.1 | 0.49 ± 0.45 |
| BERT FP32 (fine-tuned bert-base-uncased) | 0.8219 | 417.7 | 113.18 ± 71.89 |
| **BERT INT8 (ONNX Runtime, dynamic quantization)** | **0.8258** | **105.2** | **21.18 ± 11.93** |

Latency: CPU only (Intel Core i7-12700H, Windows), batch size 1, 10 warm-up + 100 timed runs,
one different test message per run. Latency budget: **50 ms per message** → met by BERT INT8 only.
Raw numbers: [`results/test_results.csv`](results/test_results.csv), data statistics: [`results/data_stats.json`](results/data_stats.json).

## How it works

```
message → regex normalization → selected model → clean / offensive + confidence → mask if offensive → chat room + latency log
          (lowercase, URLs/IPs,     (default: BERT INT8,
           leetspeak, i.d.i.o.t,     ONNX Runtime, CPU)
           repeated letters)
```

| Technique area | Where it is used |
|---|---|
| A · Neural networks | `src/models/w2v_ffn.py` – feedforward network 100→64→2, SGD + backpropagation |
| B · Classical NLP | `src/text/normalize.py` regex normalization · `src/models/lexicon_rule.py` minimum edit distance (own DP) · `src/models/tfidf_lr.py` tf-idf · word2vec trained on our data |
| C · Transformers | `src/models/train_bert.py` – fine-tuned bert-base-uncased (WordPiece tokenizer) |
| D · Efficient models | `src/models/quantize_bert.py` – ONNX export + dynamic INT8 quantization · `src/eval/final_test.py` size/latency benchmark |

## Reproducing the training

1. Accept the rules of the Kaggle competition *Jigsaw Toxic Comment Classification Challenge* and save
   your Kaggle API token to `~/.kaggle/access_token`.
2. Run, in order (GPU recommended for step 6; on Windows `setup.ps1` installs the CUDA build of PyTorch):

```bash
python -m src.data.prepare_jigsaw      # 1. fixed splits: 40k train / 5k val / 10k test
python -m src.text.normalize           # 2. shows normalization examples
python -m src.models.lexicon_rule      # 3. rule baseline
python -m src.models.tfidf_lr          # 4. tf-idf + logistic regression
python -m src.models.w2v_ffn           # 5. word2vec + feedforward network
python -m src.models.train_bert        # 6. fine-tune BERT (~26 min on an RTX 3050 Ti 4 GB)
python -m src.models.quantize_bert     # 7. ONNX + INT8
python -m src.eval.final_test          # 8. test scores, sizes, latencies → results/
```

## Data and sources

- **Jigsaw Toxic Comment Classification Challenge** (Kaggle, 2018) – dataset under CC0, comment text
  under Wikipedia's CC-BY-SA 3.0. Label rule: *offensive* if any of toxic, severe_toxic, obscene, threat,
  insult, identity_hate is 1. The data is not redistributed in this repository.
- **List of Dirty, Naughty, Obscene, and Otherwise Bad Words** (LDNOOBW, English), CC-BY 4.0 –
  `resources/bad_words_en.txt`.
- **BERT**: Devlin et al. (2019), checkpoint `bert-base-uncased` from the Hugging Face Hub.
- **word2vec**: Mikolov et al. (2013), trained with gensim on our training split.
- Quantization and benchmarking follow Tunstall, von Werra & Wolf, *Natural Language Processing with
  Transformers*, Ch. 8.

## AI usage

Claude (Anthropic) was used to help write and debug the code and draft the documentation.
All numbers in this repository come from our own runs.
