import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=64, num_heads=8):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads  # 64 // 8 = 8 dims per head
        
        # Projections for all heads (computed in parallel)
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # Output projection (combines all heads)
        self.W_o = nn.Linear(d_model, d_model)
    
    def split_heads(self, x):
        """
        Split embedding into multiple heads.
        
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, num_heads, seq_len, head_dim)
        """
        batch_size, seq_len, d_model = x.size()
        
        # Reshape: (batch, seq_len, num_heads, head_dim)
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose: (batch, num_heads, seq_len, head_dim)
        return x.transpose(1, 2)
    
    def combine_heads(self, x):
        """
        Combine multiple heads back into single embedding.
        
        Args:
            x: (batch, num_heads, seq_len, head_dim)
        Returns:
            (batch, seq_len, d_model)
        """
        batch_size, num_heads, seq_len, head_dim = x.size()
        
        # Transpose: (batch, seq_len, num_heads, head_dim)
        x = x.transpose(1, 2)
        
        # Reshape: (batch, seq_len, d_model)
        return x.contiguous().view(batch_size, seq_len, self.d_model)
    
    def attention(self, Q, K, V, mask=None):
        """
        Compute scaled dot-product attention.
        
        Args:
            Q, K, V: (batch, num_heads, seq_len, head_dim)
        Returns:
            output: (batch, num_heads, seq_len, head_dim)
            attn_weights: (batch, num_heads, seq_len, seq_len)
        """
        d_k = Q.size(-1)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        
        # Apply mask if provided (for future: causal attention)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        output = torch.matmul(attn_weights, V)
        
        return output, attn_weights
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            output: (batch, seq_len, d_model)
            attn_weights: (batch, num_heads, seq_len, seq_len)
        """
        batch_size = x.size(0)
        
        # 1. Project to Q, K, V
        Q = self.W_q(x)  # (batch, seq_len, d_model)
        K = self.W_k(x)
        V = self.W_v(x)
        
        # 2. Split into multiple heads
        Q = self.split_heads(Q)  # (batch, num_heads, seq_len, head_dim)
        K = self.split_heads(K)
        V = self.split_heads(V)
        
        # 3. Apply attention per head
        attended, attn_weights = self.attention(Q, K, V)
        # attended: (batch, num_heads, seq_len, head_dim)
        # attn_weights: (batch, num_heads, seq_len, seq_len)
        
        # 4. Combine heads
        combined = self.combine_heads(attended)  # (batch, seq_len, d_model)
        
        # 5. Final output projection
        output = self.W_o(combined)
        
        return output, attn_weights