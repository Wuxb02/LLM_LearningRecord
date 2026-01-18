import torch
import torch.nn as nn
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, d_model, d_k=None, d_v=None, dropout=0.1):
        """
        Args:
            num_heads: Number of attention heads
            d_model: Dimension of input embeddings
            d_k: Dimension of key vectors (default: d_model // num_heads)
            d_v: Dimension of value vectors (default: d_model // num_heads)
            dropout: Dropout probability
        """
        super().__init__()
        
        self.num_heads = num_heads
        self.d_model = d_model
        
        # Set default dimensions if not specified
        self.d_k = d_k if d_k is not None else d_model // num_heads
        self.d_v = d_v if d_v is not None else d_model // num_heads
        
        # Linear projections
        self.w_q = nn.Linear(d_model, num_heads * self.d_k)
        self.w_k = nn.Linear(d_model, num_heads * self.d_k)
        self.w_v = nn.Linear(d_model, num_heads * self.d_v)
        self.w_o = nn.Linear(num_heads * self.d_v, d_model)
        
        # Additional layers
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, q, k, v, mask=None):
        """
        Args:
            q: Query tensor (batch_size, seq_len_q, d_model)
            k: Key tensor (batch_size, seq_len_k, d_model)
            v: Value tensor (batch_size, seq_len_v, d_model)
            mask: Optional mask tensor (batch_size, 1, seq_len_q, seq_len_k)
            
        Returns:
            output: (batch_size, seq_len_q, d_model)
            attention_weights: (batch_size, num_heads, seq_len_q, seq_len_k)
        """
        batch_size, seq_len_q, _ = q.size()
        seq_len_k = k.size(1)
        seq_len_v = v.size(1)
        
        # Linear projections and split into heads
        q = self.w_q(q).view(batch_size, seq_len_q, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(batch_size, seq_len_v, self.num_heads, self.d_v).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask if provided
        if mask is not None:
            # Ensure mask dimensions are correct
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)  # (batch_size, 1, seq_len_q, seq_len_k)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Compute attention weights
        attention_weights = self.softmax(scores)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, v)  # (batch_size, num_heads, seq_len_q, d_v)
        
        # Concatenate heads and apply final linear projection
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len_q, -1)
        output = self.w_o(output)
        
        return output, attention_weights


# Example usage
if __name__ == "__main__":
    # Hyperparameters
    num_heads = 8
    d_model = 512
    batch_size = 4
    seq_len = 10
    
    # Create module
    mha = MultiHeadAttention(num_heads=num_heads, d_model=d_model)
    
    # Create dummy inputs
    q = torch.randn(batch_size, seq_len, d_model)
    k = torch.randn(batch_size, seq_len, d_model)
    v = torch.randn(batch_size, seq_len, d_model)
    
    # Create dummy mask (causal mask example)
    mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0)  # (1, seq_len, seq_len)
    
    # Forward pass
    output, attn_weights = mha(q, k, v, mask=mask)
    
    print("Output shape:", output.shape)  # Should be (batch_size, seq_len, d_model)
    print("Attention weights shape:", attn_weights.shape)  # Should be (batch_size, num_heads, seq_len, seq_len)