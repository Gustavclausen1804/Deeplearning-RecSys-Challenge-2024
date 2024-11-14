import torch
import torch.nn as nn
import torch.nn.functional as F

class AttLayer2(nn.Module):
    """Soft alignment attention implement.
    
    Attributes:
        dim (int): attention hidden dim
    """
    
    def __init__(self, dim=200, seed=0):
        """Initialization steps for AttLayer2.
        
        Args:
            dim (int): attention hidden dim
        """
        super().__init__()
        torch.manual_seed(seed)
        self.dim = dim
        
    def _build(self, input_shape):
        """Initialize weights if not already initialized.
        
        Args:
            input_shape (tuple): shape of input tensor
        """
        assert len(input_shape) == 3
        
        if not hasattr(self, 'W'):
            self.W = nn.Parameter(torch.empty(input_shape[-1], self.dim))
            nn.init.xavier_uniform_(self.W)
            
        if not hasattr(self, 'b'):
            self.b = nn.Parameter(torch.zeros(self.dim))
            
        if not hasattr(self, 'q'):
            self.q = nn.Parameter(torch.empty(self.dim, 1))
            nn.init.xavier_uniform_(self.q)

    def forward(self, inputs, mask=None):
        """Core implementation of soft attention.
        
        Args:
            inputs (Tensor): input tensor
            mask (Tensor, optional): input mask
            
        Returns:
            Tensor: weighted sum of input tensors
        """
        self._build(inputs.shape)
        
        attention = torch.tanh(torch.matmul(inputs, self.W) + self.b)
        attention = torch.matmul(attention, self.q)
        attention = attention.squeeze(2)
        
        if mask is None:
            attention = torch.exp(attention)
        else:
            attention = torch.exp(attention) * mask.to(inputs.dtype)
            
        attention_weight = attention / (
            attention.sum(dim=-1, keepdim=True) + torch.finfo(inputs.dtype).eps
        )
        
        attention_weight = attention_weight.unsqueeze(-1)
        weighted_input = inputs * attention_weight
        return weighted_input.sum(dim=1)


class SelfAttention(nn.Module):
    def __init__(self, multiheads, head_dim, seed=0, mask_right=False):
        super().__init__()
        torch.manual_seed(seed)
        self.multiheads = multiheads
        self.head_dim = head_dim
        self.output_dim = multiheads * head_dim
        self.mask_right = mask_right
    
    def _build(self, input_shape):
        if not hasattr(self, 'WQ'):
            self.WQ = nn.Parameter(torch.empty(input_shape[0][-1], self.output_dim))
            self.WK = nn.Parameter(torch.empty(input_shape[1][-1], self.output_dim))
            self.WV = nn.Parameter(torch.empty(input_shape[2][-1], self.output_dim))
            
            # Xavier/Glorot initialization
            nn.init.xavier_uniform_(self.WQ)
            nn.init.xavier_uniform_(self.WK)
            nn.init.xavier_uniform_(self.WV)

    def mask(self, inputs, seq_len=None, mode="add"):
        if seq_len is None:
            return inputs
            
        batch_size = inputs.shape[0]
        seq_length = inputs.shape[1]
        
        # Create one-hot encoding
        mask = torch.zeros(batch_size, seq_length, device=inputs.device)
        mask.scatter_(1, seq_len[:, 0].unsqueeze(1).long(), 1)
        mask = 1 - torch.cumsum(mask, dim=1)
        
        # Add dimensions to match input shape
        for _ in range(len(inputs.shape) - 2):
            mask = mask.unsqueeze(-1)
            
        if mode == "mul":
            return inputs * mask
        elif mode == "add":
            return inputs - (1 - mask) * 1e12

    def forward(self, QKVs):
        if len(QKVs) == 3:
            Q_seq, K_seq, V_seq = QKVs
            Q_len, V_len = None, None
        else:
            Q_seq, K_seq, V_seq, Q_len, V_len = QKVs
            
        # Build weights if not initialized
        self._build([Q_seq.shape, K_seq.shape, V_seq.shape])
        
        # Linear transformations
        Q_seq = torch.matmul(Q_seq, self.WQ).view(-1, Q_seq.size(1), self.multiheads, self.head_dim)
        K_seq = torch.matmul(K_seq, self.WK).view(-1, K_seq.size(1), self.multiheads, self.head_dim)
        V_seq = torch.matmul(V_seq, self.WV).view(-1, V_seq.size(1), self.multiheads, self.head_dim)
        
        # Permute dimensions
        Q_seq = Q_seq.permute(0, 2, 1, 3)
        K_seq = K_seq.permute(0, 2, 1, 3)
        V_seq = V_seq.permute(0, 2, 1, 3)
        
        # Attention scores
        A = torch.matmul(Q_seq, K_seq.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        
        # Apply masking
        A = A.permute(0, 3, 2, 1)
        A = self.mask(A, V_len, "add")
        A = A.permute(0, 3, 2, 1)
        
        if self.mask_right:
            ones = torch.ones_like(A[:1, :1])
            lower_triangular = torch.tril(ones)
            mask = (ones - lower_triangular) * 1e12
            A = A - mask
            
        A = F.softmax(A, dim=-1)
        
        # Output sequence
        O_seq = torch.matmul(A.transpose(-2, -1), V_seq)
        O_seq = O_seq.permute(0, 2, 1, 3)
        O_seq = O_seq.reshape(-1, O_seq.size(1), self.output_dim)
        O_seq = self.mask(O_seq, Q_len, "mul")
        
        return O_seq

class ComputeMasking(nn.Module):
    """Compute if inputs contains zero value."""
    
    def forward(self, inputs):
        mask = (inputs != 0)
        return mask.float()

class OverwriteMasking(nn.Module):
    """Set values at specific positions to zero."""
    
    def forward(self, inputs):
        return inputs[0] * inputs[1].unsqueeze(-1)

class PersonalizedAttentivePooling(nn.Module):
    """Soft alignment attention implementation."""
    
    def __init__(self, dim1, dim2, dim3, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        
        self.dropout = nn.Dropout(0.2)
        self.attention = nn.Linear(dim2, dim3)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.attention.weight)
        nn.init.zeros_(self.attention.bias)

    def forward(self, vecs_input, query_input):
        user_vecs = self.dropout(vecs_input)
        user_att = self.tanh(self.attention(user_vecs))
        
        # Dot product with query
        user_att2 = torch.sum(user_att * query_input.unsqueeze(1), dim=-1)
        user_att2 = self.softmax(user_att2)
        
        # Final dot product
        user_vec = torch.sum(user_vecs * user_att2.unsqueeze(-1), dim=1)
        return user_vec
