import torch
import torch.nn as nn
import torch.nn.functional as F


# Ensure that AttLayer2 outputs a tensor with the correct final dimension
class AttLayer2(nn.Module):
    """Soft alignment attention implementation."""

    def __init__(self, dim=200, seed=0, device='cuda'):
        super(AttLayer2, self).__init__()
        self.dim = dim
        self.device = device
        torch.manual_seed(seed)
        self.W = None  # Will be initialized in build()
        self.b = nn.Parameter(torch.zeros(dim).to(device))
        self.q = nn.Linear(in_features=dim, out_features=1, bias=False).to(device)
    
    def build(self, input_shape):
        in_features = input_shape[-1]
        self.W = nn.Linear(in_features=in_features, out_features=self.dim, bias=True).to(self.device)
        #print(f"AttLayer2 build: W initialized with in_features={in_features}, out_features={self.dim}")
    
    def forward(self, inputs, mask=None):
        inputs = inputs.to(self.device)
        if self.W is None:
            self.build(inputs.shape)
        
        # Transform inputs
        h = torch.tanh(self.W(inputs) + self.b)  # Shape: (batch_size, seq_length, self.dim)
        #print(f"AttLayer2 Forward - transformed inputs h shape: {h.shape}")
        
        # Compute attention scores
        attention = self.q(h).squeeze(-1)  # Shape: (batch_size, seq_length)
        #print(f"AttLayer2 Forward - attention scores shape: {attention.shape}")
        
        if mask is not None:
            mask = mask.to(self.device)
            attention = attention.masked_fill(~mask, float('-inf'))
        
        # Compute attention weights
        attention_weight = F.softmax(attention, dim=-1).unsqueeze(-1)  # Shape: (batch_size, seq_length, 1)
        #print(f"AttLayer2 Forward - attention weights shape: {attention_weight.shape}")
        
        # Apply attention weights to the transformed inputs
        weighted_input = h * attention_weight  # Shape: (batch_size, seq_length, self.dim)
        #print(f"AttLayer2 Forward - weighted input shape: {weighted_input.shape}")
        
        # Sum over the sequence length dimension
        output = weighted_input.sum(dim=1)  # Shape: (batch_size, self.dim)
        #print(f"AttLayer2 Forward - output shape after summing over sequence length: {output.shape}")
        
        return output





class SelfAttention(nn.Module):
    """Multi-head self-attention implementation."""

    def __init__(self, multiheads, head_dim, seed=0, mask_right=False, device='cuda', **kwargs):
        """Initialization steps for SelfAttention."""
        super(SelfAttention, self).__init__(**kwargs)
        self.multiheads = multiheads
        self.head_dim = head_dim
        self.output_dim = multiheads * head_dim
        self.mask_right = mask_right
        self.device = device
        torch.manual_seed(seed)

    def build(self, input_shapes):
        """Initialization for variables in SelfAttention."""
        Q_shape, K_shape, V_shape = input_shapes[:3]
        in_features_q = Q_shape[-1]
        in_features_k = K_shape[-1]
        in_features_v = V_shape[-1]

        self.WQ = nn.Linear(in_features=in_features_q, out_features=self.output_dim, bias=False).to(self.device)
        self.WK = nn.Linear(in_features=in_features_k, out_features=self.output_dim, bias=False).to(self.device)
        self.WV = nn.Linear(in_features=in_features_v, out_features=self.output_dim, bias=False).to(self.device)
        #print(f"SelfAttention build: WQ, WK, WV initialized with in_features={in_features_q}, {in_features_k}, {in_features_v} and out_features={self.output_dim}")

    def forward(self, QKVs):
        """Core logic of multi-head self-attention."""
        if len(QKVs) == 3:
            Q_seq, K_seq, V_seq = [q.to(self.device) for q in QKVs]
        elif len(QKVs) == 5:
            Q_seq, K_seq, V_seq, Q_len, V_len = [q.to(self.device) if isinstance(q, torch.Tensor) else q for q in QKVs]
        else:
            raise ValueError("QKVs must be a list of 3 or 5 tensors.")

        if not hasattr(self, 'WQ'):
            self.build([Q_seq.shape, K_seq.shape, V_seq.shape])

        #print("SelfAttention Forward - Q_seq shape:", Q_seq.shape)
        #print("SelfAttention Forward - K_seq shape:", K_seq.shape)
        #print("SelfAttention Forward - V_seq shape:", V_seq.shape)

        Q = self.WQ(Q_seq)
        K = self.WK(K_seq)
        V = self.WV(V_seq)
        #print("SelfAttention Forward - Q, K, V shapes after linear transformations:", Q.shape, K.shape, V.shape)

        batch_size = Q.size(0)

        Q = Q.view(batch_size, -1, self.multiheads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.multiheads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.multiheads, self.head_dim).transpose(1, 2)
        #print("SelfAttention Forward - Q, K, V shapes after view and transpose:", Q.shape, K.shape, V.shape)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        #print("SelfAttention Forward - scores shape:", scores.shape)

        attn = F.softmax(scores, dim=-1)
        attn = F.dropout(attn, p=0.1, training=self.training)
        #print("SelfAttention Forward - attention shape after softmax:", attn.shape)

        output = torch.matmul(attn, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.output_dim)
        #print("SelfAttention Forward - output shape after concatenation:", output.shape)

        return output


class ComputeMasking(nn.Module):
    """Compute if inputs contain zero value."""
    def forward(self, inputs):
        """Compute mask where inputs are not equal to zero."""
        #print("ComputeMasking Forward - inputs shape:", inputs.shape)
        mask = inputs != 0
        #print("ComputeMasking Forward - mask shape:", mask.shape)
        return mask.float()


class OverwriteMasking(nn.Module):
    """Set values at specific positions to zero based on mask."""
    def forward(self, inputs):
        """Apply mask to inputs."""
        #print("OverwriteMasking Forward - value tensor shape:", inputs[0].shape)
        #print("OverwriteMasking Forward - mask tensor shape:", inputs[1].shape)
        return inputs[0] * inputs[1].unsqueeze(-1)


def PersonalizedAttentivePooling(dim1, dim2, dim3, seed=0):
    """Soft alignment attention implementation."""
    class PersonalizedAttentivePoolingModule(nn.Module):
        def __init__(self, dim1, dim2, dim3, seed):
            super(PersonalizedAttentivePoolingModule, self).__init__()
            torch.manual_seed(seed)
            self.dropout = nn.Dropout(p=0.2)
            self.dense = nn.Linear(in_features=dim2, out_features=dim3)
            self.activation = nn.Tanh()
            self.dot = nn.Linear(in_features=dim3, out_features=1, bias=False)
            self.softmax = nn.Softmax(dim=1)

        def forward(self, vecs_input, query_input):
            #print("PersonalizedAttentivePooling Forward - vecs_input shape:", vecs_input.shape)
            #print("PersonalizedAttentivePooling Forward - query_input shape:", query_input.shape)
            user_vecs = self.dropout(vecs_input)
            user_att = self.activation(self.dense(user_vecs))
            #print("PersonalizedAttentivePooling Forward - user_att shape after dense and activation:", user_att.shape)

            user_att2 = torch.matmul(user_att, query_input.unsqueeze(-1)).squeeze(-1)
            user_att2 = self.softmax(user_att2).unsqueeze(1)
            #print("PersonalizedAttentivePooling Forward - user_att2 shape after softmax and unsqueeze:", user_att2.shape)

            user_vec = torch.bmm(user_att2, user_vecs).squeeze(1)
            #print("PersonalizedAttentivePooling Forward - user_vec shape after batch matrix multiplication:", user_vec.shape)
            return user_vec

    return PersonalizedAttentivePoolingModule(dim1, dim2, dim3, seed)
