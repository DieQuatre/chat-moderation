| Model | Test macro-F1 | Size (MB) | Latency (ms, mean ± std) |
|---|---|---|---|
| Lexicon rule | 0.7294 | 0.2 | 0.15 ± 0.20 |
| tf-idf + LR | 0.7795 | 4.6 | 1.12 ± 0.43 |
| word2vec + FFN | 0.7685 | 30.1 | 0.49 ± 0.45 |
| BERT FP32 | 0.8219 | 417.7 | 113.18 ± 71.89 |
| BERT INT8 | 0.8258 | 105.2 | 21.18 ± 11.93 |

Latency: 10 warm-up + 100 timed runs, Intel64 Family 6 Model 154 Stepping 3, GenuineIntel (Windows), CPU only, batch size 1.