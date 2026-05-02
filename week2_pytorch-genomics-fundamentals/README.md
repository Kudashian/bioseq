# DNA Sequence Classifier with PyTorch

A neural network that classifies DNA sequences as coding or non-coding based on GC content patterns.

## 🎯 Project Overview

This project demonstrates PyTorch fundamentals by building a binary classifier for DNA sequences. The model learns to distinguish between coding regions (high GC content) and non-coding regions (low GC content).

## 📊 Results

- **Training Accuracy:** 100.00%
- **Test Accuracy:** 100.00%
- **Final Training Loss:** 0.0823
- **Final Test Loss:** 0.0956

![Training Curves](training_curves.png)
![Confusion Matrix](confusion_matrix.png)

## 🏗️ Architecture

**SimpleDNAClassifier:**

Input (40 features: 10 nucleotides × 4 one-hot)
  ↓
Linear Layer (40 → 64)
  ↓
ReLU Activation
  ↓
Linear Layer (64 → 1)
  ↓
Sigmoid Activation
  ↓
Output (probability 0-1)


**Total Parameters:** 2,689
- fc1: 2,560 weights + 64 biases
- fc2: 64 weights + 1 bias

## 🔧 Technical Details

**Data Generation:**
- 1,000 synthetic sequences (500 coding, 500 non-coding)
- Coding sequences: 70% GC content
- Non-coding sequences: 30% GC content
- Sequence length: 10 nucleotides

**Training Configuration:**
- Loss Function: Binary Cross Entropy (BCELoss)
- Optimizer: Adam (lr=0.001)
- Epochs: 100
- Train/Test Split: 80/20

**Technologies:**
- PyTorch 2.x
- Python 3.11+
- scikit-learn (train/test split, metrics)
- matplotlib/seaborn (visualization)

## 📁 Project Structure

pytorch-genomics-fundamentals/
├── notebooks/
│   ├── 01_tensor_basics.ipynb
│   ├── 02_autograd_exploration.ipynb
│   ├── 03_building_neural_networks.ipynb
│   └── 04_training_dna_classifier.ipynb
├── src/
│   ├── __init__.py
│   ├── sequence_encoder.py    # One-hot encoding for DNA
│   └── models.py               # Neural network architecture
├── saved_models/
│   └── dna_classifier.pth      # Trained model
├── training_curves.png
├── confusion_matrix.png
├── README.md
└── requirements.txt


## 🚀 Usage

### Installation
```bash
# Clone repository
git clone [your-repo-url]
cd pytorch-genomics-fundamentals

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 📚 What I Learned

### Week 2 - PyTorch Fundamentals

**Day 1:** Tensor operations and manipulation
- Creating tensors, shapes, indexing
- Tensor operations vs NumPy arrays
- GPU acceleration concepts

**Day 2:** Sequence encoding
- One-hot encoding for biological sequences
- Custom encoder class with encoding/decoding
- Data representation for neural networks

**Day 3:** Autograd and gradient descent
- Automatic differentiation
- Understanding gradients and backpropagation
- Manual vs automated gradient descent
- Why we subtract gradients (optimization)

**Day 4:** Building neural networks
- `nn.Module` pattern
- Linear layers and activations
- Forward pass architecture
- Parameter counting

**Day 5:** Training loop
- Loss functions (BCELoss)
- Optimizers (Adam)
- Training vs evaluation modes
- Accuracy metrics

**Day 6-7:** Evaluation and deployment
- Confusion matrices
- Precision, Recall, F1-Score
- Model saving/loading
- Documentation

## 🎓 Key Concepts Demonstrated

1. **Tensor Operations:** Shape manipulation, dtype handling
2. **Data Encoding:** Converting categorical biological data to numerical format
3. **Neural Network Architecture:** Designing layers for classification
4. **Gradient Descent:** Understanding optimization at fundamental level
5. **Training Pipeline:** Complete ML workflow from data to deployment
6. **Model Evaluation:** Proper metrics beyond simple accuracy

## 🔮 Future Improvements

- [ ] Handle variable-length sequences (padding/RNNs)
- [ ] Use real genomic data (NCBI datasets)
- [ ] Implement CNN for position-independent patterns
- [ ] Add attention mechanisms
- [ ] Multi-class classification (coding/non-coding/regulatory)
- [ ] Cross-validation for robust evaluation

## 📖 References

- PyTorch Documentation: https://pytorch.org/docs/
- Biological sequence analysis concepts
- Week 2 learning progression from PyTorch fundamentals course

## 👤 Author

[Kudakwashe Nyamupangedengu]
- Learning Journey: Genomics → Healthcare AI
- Background: Bioinformatics, NGS pipelines
- Goal: Transition to AI/ML roles in healthcare/biotech

## 📄 License

This project is for educational purposes.

---

**Status:** ✅ Week 2 Complete - PyTorch Fundamentals Mastered
**Next:** Week 3 - Deep Learning Architectures & Real Data