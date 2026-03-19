# Week 6: RNNs and LSTMs

## Overview

This week focused on recurrent architectures — building a vanilla RNN and a full LSTM cell from scratch, training on real genomic data, and critically evaluating when these architectures are and aren't appropriate.

---

## What Was Built

### 1. Vanilla RNN Cell (`VanillaRNN.py`)
Built from scratch using two `nn.Linear` layers — one for the input (`W_xh`) and one for the hidden state (`W_hh`):

```
h_t = tanh(W_xh(x_t) + W_hh(h_prev))
```

Looped manually over one-hot encoded DNA sequences (4 nucleotides × 10 timesteps) to confirm the hidden state updates correctly at each timestep.

---

### 2. LSTM Cell from Scratch (`VanillaLSTM.py`)
Implemented all four gates manually with separate weight matrices for input and hidden connections:

| Gate | Function |
|------|----------|
| Forget (`f_t`) | Decides what to remove from cell state |
| Input (`i_t`) | Decides what new information to write |
| Candidate (`g_t`) | New candidate values to add |
| Output (`o_t`) | Decides what part of cell state to expose as `h_t` |

```python
f_t = sigmoid(W_fh(x_t) + W_fhh(h_prev))
i_t = sigmoid(W_ih(x_t) + W_ihh(h_prev))
g_t = tanh(W_gh(x_t) + W_ghh(h_prev))
o_t = sigmoid(W_oh(x_t) + W_ohh(h_prev))

c_t = f_t * c_prev + i_t * g_t
h_t = o_t * tanh(c_t)
```

---

### 3. LSTM Classifier (`LSTMTrainer.py`)
Extended the LSTM cell into a full binary classifier for the UCI Promoter dataset (106 samples, 57bp sequences):

- Loops over each nucleotide timestep, updating `c_prev` and `h_prev`
- Applies a linear classification head to the final `h_t`
- Trained with `BCELoss` and `Adam` optimizer

**Best single run accuracy: ~86%**

---

### 4. nn.LSTM Comparison (`nn_LSTM.py`)
Swapped the custom cell for PyTorch's optimised `nn.LSTM`, processing the full sequence in one call. Used as a validation baseline.

---

### 5. K-Fold Cross Validation
Applied 5-fold stratified CV to both models to get stable accuracy estimates:

| Model | Mean Accuracy | Std Dev |
|-------|--------------|---------|
| Custom LSTM | ~54% | ~8% |
| nn.LSTM | ~54% | ~8% |

---

## Key Findings

**The LSTM is the wrong tool for this dataset.** Two reasons:

1. **Dataset too small** — 106 samples is insufficient for an LSTM to learn meaningful sequential patterns. High variance between runs confirmed this.

2. **Sequences too short** — At 57bp, promoter recognition is driven by local motifs (TATA box ~6bp, CAAT box ~9bp). A CNN with appropriately-sized kernels captures this better. LSTM's strength is long-range dependencies, which simply don't exist at this scale.

The Week 4 CNN/MLP achieved ~97% on this same dataset — confirming that architectural fit matters as much as model complexity.

---

## Key Concepts Learned

- **Vanishing gradients** — repeated multiplication through `W_hh` causes gradient signal from early timesteps to decay exponentially during backprop
- **LSTM cell state** — acts as long-term memory, bypassing the vanishing gradient problem via additive updates
- **Gate mechanics** — forget, input, output gates control information flow; each has independent weights
- **State management** — `c_prev` and `h_prev` must be updated each timestep and reset between independent sequences
- **Dead gradients** — identified and debugged zero-gradient hidden weights caused by missing state updates in the forward loop
- **Architecture selection** — LSTM is appropriate for long sequences with long-range dependencies; CNNs are preferred for short sequences with local motif patterns

---

## Debugging Log

| Bug | Symptom | Fix |
|-----|---------|-----|
| Shared weights across gates | Model not learning | Separate `nn.Linear` per gate |
| `h_prev` not updated in loop | Dead hidden gradients | Add `h_prev = h_t` inside timestep loop |
| Accuracy >100% | `total += y_test.sum()` | Changed to `total += 1` |
| Single prediction for 22 sequences | Shape mismatch | Confirmed per-sequence loop, fixed transpose |

---

## Files
```
week6_RNNs/
├── src/
│   ├── SequenceEncoder.py
│   ├── UpdatedSequenceEncoder.py
│   ├── VanillaRNN.py
│   ├── VanillaLSTM.py
│   ├── LSTMTrainer.py
│   └── LSTMClassifier.py
└── notebooks/
    ├── 01_simpleRNN.ipynb
    ├── 02_simpleLSTM.ipynb
    ├── 03_LSTMClassifier.ipynb
    └── 04_LSTMwithKFold.ipynb
```

---

## Next: Week 7
- Sequence-to-sequence models
- Encoder-decoder architecture
- Attention mechanisms in context of seq2seq
