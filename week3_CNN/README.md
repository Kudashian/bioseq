# Week 3: CNNs for Genomic Sequence Classification

## Overview

This week extends the Week 2 PyTorch fundamentals by replacing the simple linear classifier with a 2-layer CNN architecture capable of handling variable-length DNA sequences. The model learns to classify sequences as coding (high GC content) or non-coding (low GC content) using biologically motivated convolutional filters.

---

## What I Built

### 1. DataLoaders & Batching
Refactored the data pipeline from raw tensor operations to PyTorch's `Dataset` and `DataLoader` abstractions, enabling batched training with shuffling.

### 2. 2-Layer CNN Architecture
Replaced the Week 2 linear classifier with a hierarchical CNN:

```
Input:          [batch, 4, seq_length]    ← one-hot encoded DNA (4 bases)
Conv1d Layer 1: [batch, 64, seq_length]   ← 64 basic motif detectors
ReLU
Conv1d Layer 2: [batch, 128, seq_length]  ← 128 motif combination detectors
ReLU
Global Avg Pool:[batch, 128, 1]           ← position-invariant features
Squeeze:        [batch, 128]
Linear:         [batch, 1]                ← binary classification
Sigmoid:        probability (0-1)
```

**Total parameters:** significantly more than Week 2's 2,689 — reflecting the increased capacity to detect sequence motifs.

### 3. Variable-Length Sequence Handling
Implemented global average pooling (`nn.AdaptiveAvgPool1d`) to decouple the architecture from fixed sequence lengths. A custom `collate_fn` pads sequences within each batch to the longest sequence in that batch.

```python
# Custom collate function pads within each batch
def collate_variable_length(batch):
    sequences, labels = zip(*batch)
    max_len = max(seq.shape[-1] for seq in sequences)
    # Pad shorter sequences with zeros
    ...
    return torch.stack(padded), torch.stack(list(labels))
```

### 4. Improved Synthetic Dataset
Upgraded from 10bp fixed-length sequences to a more realistic 3-class synthetic dataset:

| Class | GC Content | Label |
|-------|-----------|-------|
| High GC (coding) | >60% | 1 |
| Low GC (non-coding) | <40% | 0 |
| Boundary | ~50% ± 5% | 0 or 1 by threshold |

Sequence lengths randomly sampled from: 50bp, 100bp, 150bp, 200bp, 250bp.

---

## Results

| Metric | Value |
|--------|-------|
| Train Accuracy | 100% |
| Test Accuracy | 98% |
| False Positives | 4 (non-coding → coding) |
| False Negatives | 0 |

### Training Curves
The gap between train loss (→0) and test loss (~0.04) indicates mild overfitting — a known limitation of synthetic data with a strong GC signal. Interested to see how real genomic data will go

### Confusion Matrix
```
                Predicted
                Non-coding  Coding
Actual Non-cod      97         4
       Coding         0        99
```

---

## Key Concepts Learned

**Conv1d shape requirements:** `[batch, channels, length]` — channels represent base identity (A/C/G/T), length is sequence position. Flattening before convolution destroys positional information.

**Hierarchical feature detection:** Layer 1 detects individual motifs (e.g. TATA box, GC-rich regions). Layer 2 detects combinations of those motifs — analogous to how regulatory modules combine transcription factor binding sites.

**Global pooling vs flattening:** Flattening hardcodes sequence length into the architecture (`fc = Linear(128 × seq_len, 1)`). Global pooling compresses any sequence length to a fixed-size feature vector (`[batch, 128]`), enabling the same trained model to process sequences of any length at inference time.

**Batching variable-length sequences:** PyTorch requires all tensors in a batch to have identical shapes. The solution is to pad within each batch to the longest sequence in that batch — not across the entire dataset.

---

## Architecture Decision: Why Global Pooling?

```
Without global pooling:
  Model trained on 100bp → FAILS on 50bp or 250bp sequences
  fc layer hardcoded to expect 128 × 100 = 12,800 inputs

With global pooling:
  Same trained model handles 50bp, 100bp, 5000bp sequences
  fc layer always receives 128 features regardless of length
```

For genomics tasks, this matters — real promoter sequences, enhancers, and coding regions vary significantly in length across species and databases.

---

## Project Structure

```
week3_cnn_genomics/
├── notebooks/
│   ├── 01_dataloaders_batching.ipynb      ← Day 1-2
│   ├── 02_cnn_architecture.ipynb          ← Day 3-4
│   └── 03_variable_length_evaluation.ipynb ← Day 5-7
├── src/
│   ├── models.py                          ← SimpleDNAClassifier (CNN)
│   ├── sequence_encoder.py                ← One-hot encoding (from Week 2)
│   └── data_utils.py                      ← DNADataset, collate_fn
├── saved_models/
│   └── cnn_classifier.pth
├── training_curves.png
├── confusion_matrix.png
└── README.md
```

## Next: Week 4 — Real Genomic Data

Moving from synthetic GC content classification to real promoter/non-promoter sequences from NCBI/Ensembl. Key challenges: class imbalance, noisy labels, longer sequences, real biological complexity.
