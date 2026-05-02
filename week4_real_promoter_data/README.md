# Promoter CNN Classifier

A 1D Convolutional Neural Network for classifying DNA promoter 
sequences using real genomic data.

## 🎯 Project Overview

Classifies DNA sequences as promoter or non-promoter using the 
UCI Promoter Gene Sequences dataset (106 real biological sequences).
Demonstrating practises including K-Fold cross 
validation, regularization, and experiment tracking.

## 📊 Results

| Model | Mean Accuracy | Std Dev |
|-------|--------------|---------|
| BatchNorm + Dropout | **97.14%** | 3.81% |
| Baseline CNN | 96.23% | 3.55% |
| Small + Dropout(0.6) | 96.23% | 3.55% |
| Dropout(0.5) only | 93.42% | 4.81% |

**Best model:** 2-layer CNN with BatchNorm and Dropout

## 🏗️ Architecture
```
Input: [batch, 4, 57]  ← one-hot encoded, 57bp sequences
  ↓
Conv1d(4, 96, kernel=10) + BatchNorm + ReLU
  ↓
Conv1d(96, 192, kernel=10) + BatchNorm + ReLU  
  ↓
AdaptiveMaxPool1d(1)  ← handles variable-length sequences
  ↓
Dropout(0.5)
  ↓
Linear(192, 1) + Sigmoid
  ↓
Output: probability (promoter vs non-promoter)
```

**Design decisions:**
- kernel_size=10 matches biological promoter motif lengths
  (TATA box ~6bp, GC box ~6bp, CAAT box ~5bp)
- Global max pooling: "does this motif exist anywhere?"
- BatchNorm before ReLU: stabilizes activation distributions

## 🔬 Dataset

**UCI Promoter Gene Sequences**
- 106 real E. coli DNA sequences
- 57 nucleotides each
- Balanced: 53 promoters (+), 53 non-promoters (-)
- Source: UCI ML Repository

**Preprocessing:**
- Strip whitespace, convert to uppercase
- Unknown nucleotides (N) encoded as zero vector
- One-hot encoding: [A, C, G, T] → [1,0,0,0], [0,1,0,0], etc.

## 📈 Evaluation

**5-Fold Stratified Cross Validation:**
- Each fold: ~85 training, ~21 test samples
- Every sequence tested exactly once
- Stratified: maintains 50/50 class balance per fold
- Fresh model per fold prevents data leakage

**Why K-Fold?** With only 106 samples, a single train/test
split gives unreliable estimates (1 wrong = 5% accuracy change).

## 🔧 Regularization Analysis

**Overfitting detected in baseline:**
```
Train loss: 0.0008  (near perfect)
Test loss:  0.2052  (250x larger!)

Noted that metrics inflated by tiny dataset.
```

**Solutions applied:**
- Dropout(0.5): randomly disables neurons during training
  → Forces independent feature learning
  → Disabled during evaluation (uses all neurons)
- BatchNorm: normalizes activations to μ=0, σ=1
  → Placed before ReLU for balanced activations
  → Uses running averages during inference

## 🗂️ Project Structure
```
week4-promoter-cnn/
├── notebooks/
│   └── 05_real_promoter_data.ipynb  # Full pipeline
├── src/
│   ├── sequence_encoder.py          # One-hot encoding + N handling
│   └── promoter_cnn.py              # CNN architecture
└── README.md
```

## 🚀 Usage
```python
from src.promoter_cnn import PromoterCNNClassifier
from src.sequence_encoder import SequenceEncoder
import torch

# Load model
model = PromoterCNNClassifier(4, 96, 1, dropout_p=0.5)
checkpoint = torch.load('saved_models/promoter_cnn.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Classify sequence
encoder = SequenceEncoder()
sequence = "TACTAGCAATACGCTTGCGTTCGGTGGTTAAGTATGTATAATGCGCGGGCTTGTCGT"
encoded = encoder.encode(sequence).T.unsqueeze(0)  # [1, 4, 57]

with torch.no_grad():
    prob = model(encoded).item()
    print(f"Promoter probability: {prob:.2%}")
```

## 📚 Key Learnings

**Week 4 concepts demonstrated:**
- Real genomic data preprocessing
- 1D CNN architecture for biological sequences  
- K-Fold cross validation for small datasets
- Dropout and BatchNorm regularization
- Experiment tracking (manual → wandb)
- Overfitting detection and mitigation

## 🔮 Future Improvements

- [ ] Larger dataset
- [ ] Attention mechanism for position-aware detection
- [ ] Transfer learning from pre-trained DNA models
- [ ] Multi-species promoter prediction
- [ ] Interpretability: which motifs drive predictions?

## 👤 Author

**Kuda Nyamupa** | Genomics → Healthcare AI Transition  
Background: Bioinformatics, NGS pipelines, Production AI Systems  
Learning: PyTorch → Deep Learning → MLOps

---
**Week 4 Complete** ✅ | Next: Transformers & Attention Mechanisms