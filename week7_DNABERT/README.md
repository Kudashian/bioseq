# Week 7: Transfer Learning — DNABERT Fine-tuning vs Nucleotide Transformer
**Project Butterfly — ML Engineering Curriculum**
**GitHub:** Kudashian

---

## Overview

This week focused on fine-tuning pretrained DNA language models on the UCI Promoter dataset, comparing performance against the custom architectures built in Weeks 4–6. The core question: does a large pretrained Transformer outperform simpler architectures on a small genomics dataset?

---

## What Was Built

### 1. DNABERT Fine-tuning (`dnabert_finetuning.ipynb`)
Fine-tuned `zhihan1996/DNA_bert_6` (110M parameters, 12-layer BERT) for binary promoter classification.

**Key implementation details:**
- DNA sequences converted to space-separated 6-mers before tokenization
- HuggingFace `AutoModelForSequenceClassification` with `num_labels=2`
- AdamW optimizer with `lr=2e-5` (critical — default `1e-3` destroyed pretrained weights)
- 5-fold stratified cross-validation on 106 samples
- Kaggle P100 GPU

---

## K-mer Tokenization

DNABERT expects sequences as space-separated k-mers:

```python
def sequence_to_kmer(sequence, k=6):
    kmer_list = []
    for i in range(len(sequence) - k + 1):
        kmer_list.append(sequence[i:i+k])
    return " ".join(kmer_list)

# ATCGGA → "ATCGGA" (single 6-mer for 6bp sequence)
# Longer: ATCGGAT → "ATCGGA TCGGAT"
```

Note: `range(len(sequence) - k + 1)` — the `+1` is critical to include the last valid k-mer.

---

## Key Debugging Journey

| Bug | Symptom | Fix |
|-----|---------|-----|
| Learning rate too high (1e-3) | Identical logits across all samples, model collapsed | Changed to `lr=2e-5` |
| `np.where` label lookup | Loss of ~13 per sample, impossible for 2-class problem | Switched to `enumerate` for index-based label access |
| Labels not tensors | `AttributeError: 'int' has no attribute dtype` | `torch.tensor([label])` |
| Loss accumulation | Final loss not dividing correctly | `final_loss += outputs.loss.item()` |
| Model not reset per fold | Weights bleeding across folds | Reinitialise model inside fold loop |

**Key diagnostic tool — logit inspection:**
- Identical logits → collapsed model (learning rate or class imbalance issue)
- Logits near zero → model uncertain, hasn't learned (too few samples/epochs)
- Logits varying meaningfully → model discriminating correctly

---

## Results

### Cross-Validation Comparison

| Model | Mean CV Accuracy | Notes |
|-------|-----------------|-------|
| CNN (Week 4) | ~97% | 5-fold, local |
| Custom LSTM (Week 6) | ~54% | 5-fold, local |
| DNABERT fine-tuned (Week 7) | ~78% | 3-fold, Kaggle P100, 10 epochs |

Training loss: **0.058** vs Test loss: **0.429** — significant gap indicating overfitting on small dataset.

---

## Key Findings

**Bigger is not always better.** DNABERT's 110M parameters are poorly matched to 106 training samples. The model memorises the training fold (loss → 0.016) but fails to generalise.

**Learning rate is critical for fine-tuning.** Using the default Adam lr of 1e-3 catastrophically destroyed pretrained weights — all logits collapsed to identical values. Fine-tuning requires lr ~2e-5 to preserve pretrained knowledge while adapting the classification head.

**Distribution mismatch matters.** DNABERT was pretrained on human genome data. Bacterial promoter sequences follow different statistical patterns, limiting transfer effectiveness.

**Architecture-task matching remains the dominant factor:**
- CNN wins because promoter signals (TATA box, CAAT box) are local motifs — exactly what convolution captures
- LSTM underperforms because 57bp sequences have no long-range dependencies to learn
- DNABERT is limited by dataset size — would likely excel with thousands of sequences

---

## Biological Interpretation

DNABERT's k-mer vocabulary implicitly encodes biological grammar. A single nucleotide ('A') carries no functional meaning, but 'TATAAT' is a TATA box — a transcription initiation signal. K-mer tokenization allows the model to learn co-occurrence patterns analogous to masked language modelling in NLP, without explicit biological annotation.

This is why DNABERT transfers well to human genomics tasks but struggles on bacterial data with only 106 samples — insufficient signal to overcome the domain gap.

---

## Next: Week 8
- MLOps — experiment tracking with Weights & Biases
- Proper model versioning and reproducible training runs
- Deploying a model as an API endpoint
