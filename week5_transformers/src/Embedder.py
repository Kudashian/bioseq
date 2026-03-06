import torch
import torch.nn as nn
import math

class DNAEmbedding(nn.Module):
    def __init__(self, vocab_size=4, d_model=64):
        """
        Transform one-hot vectors into richer embeddings.
        
        Args:
            vocab_size: 4 for DNA (A, C, G, T)
            d_model: embedding dimension (64, 128, 256, etc.)
        """
        super().__init__()
        self.embedding = nn.Linear(vocab_size, d_model)
        self.d_model = d_model
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, 4) one-hot encoded
        Returns:
            (batch, seq_len, d_model) embeddings
        """
        # Scale by sqrt(d_model) - standard practice in Transformers
        return self.embedding(x) * math.sqrt(self.d_model)