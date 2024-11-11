import torch
import torch.nn as nn
import torch.nn.functional as F


class AttLayer2(nn.Module):
    """Soft alignment attention implementation.

    Attributes:
        dim (int): Attention hidden dimension.
    """

    def __init__(self, dim=200, seed=0):
        """Initialization steps for AttLayer2.

        Args:
            dim (int): Attention hidden dimension.
            seed (int): Random seed for reproducibility.
        """
        super(AttLayer2, self).__init__()
        self.dim = dim
        torch.manual_seed(seed)
        self.W = nn.Linear(in_features=0, out_features=dim, bias=True)  # Placeholder, will set in build
        self.b = nn.Parameter(torch.zeros(dim))
        self.q = nn.Linear(in_features=dim, out_features=1, bias=False)

    def build(self, input_shape):
        """Initialization for variables in AttLayer2.
        There are three variables in AttLayer2, i.e., W, b, and q.

        Args:
            input_shape (torch.Size): Shape of input tensor.
        """
        # Adjust the input features based on input_shape
        in_features = input_shape[-1]
        self.W = nn.Linear(in_features=in_features, out_features=self.dim, bias=True)
        self.q = nn.Linear(in_features=self.dim, out_features=1, bias=False)

    def forward(self, inputs, mask=None):
        """Core implementation of soft attention.

        Args:
            inputs (torch.Tensor): Input tensor of shape (batch_size, seq_length, input_dim).
            mask (torch.Tensor, optional): Mask tensor of shape (batch_size, seq_length). Defaults to None.

        Returns:
            torch.Tensor: Weighted sum of input tensors of shape (batch_size, input_dim).
        """
        if not hasattr(self, 'W'):
            self.build(inputs.shape)

        # Apply linear transformation and tanh activation
        attention = torch.tanh(self.W(inputs) + self.b)  # (batch_size, seq_length, dim)
        attention = self.q(attention).squeeze(-1)  # (batch_size, seq_length)

        if mask is not None:
            attention = attention.masked_fill(~mask, float('-inf'))

        attention_weight = F.softmax(attention, dim=-1)  # (batch_size, seq_length)
        attention_weight = attention_weight.unsqueeze(-1)  # (batch_size, seq_length, 1)

        weighted_input = inputs * attention_weight  # (batch_size, seq_length, input_dim)
        return weighted_input.sum(dim=1)  # (batch_size, input_dim)


class SelfAttention(nn.Module):
    """Multi-head self-attention implementation.

    Args:
        multiheads (int): The number of heads.
        head_dim (int): Dimension of each head.
        mask_right (bool): Whether to mask future positions (causal mask).
        seed (int): Random seed for reproducibility.

    Returns:
        torch.Tensor: Weighted sum after attention.
    """

    def __init__(self, multiheads, head_dim, seed=0, mask_right=False):
        """Initialization steps for SelfAttention.

        Args:
            multiheads (int): The number of heads.
            head_dim (int): Dimension of each head.
            mask_right (bool): Whether to mask future positions (causal mask).
            seed (int): Random seed for reproducibility.
        """
        super(SelfAttention, self).__init__()
        self.multiheads = multiheads
        self.head_dim = head_dim
        self.output_dim = multiheads * head_dim
        self.mask_right = mask_right
        torch.manual_seed(seed)

        self.WQ = nn.Linear(in_features=0, out_features=self.output_dim, bias=False)  # Placeholder
        self.WK = nn.Linear(in_features=0, out_features=self.output_dim, bias=False)  # Placeholder
        self.WV = nn.Linear(in_features=0, out_features=self.output_dim, bias=False)  # Placeholder

    def build(self, input_shapes):
        """Initialization for variables in SelfAttention.
        There are three variables in SelfAttention, i.e., WQ, WK, and WV.

        Args:
            input_shapes (tuple of torch.Size): Shapes of query, key, and value tensors.
        """
        # Assume input_shapes is a tuple: (Q_shape, K_shape, V_shape)
        Q_shape, K_shape, V_shape = input_shapes[:3]
        in_features_q = Q_shape[-1]
        in_features_k = K_shape[-1]
        in_features_v = V_shape[-1]

        self.WQ = nn.Linear(in_features=in_features_q, out_features=self.output_dim, bias=False)
        self.WK = nn.Linear(in_features=in_features_k, out_features=self.output_dim, bias=False)
        self.WV = nn.Linear(in_features=in_features_v, out_features=self.output_dim, bias=False)

    def forward(self, QKVs):
        """Core logic of multi-head self-attention.

        Args:
            QKVs (list of torch.Tensor): Inputs of multi-head self-attention, i.e., [Q, K, V] or [Q, K, V, Q_len, V_len].

        Returns:
            torch.Tensor: Output tensors after attention.
        """
        if len(QKVs) == 3:
            Q_seq, K_seq, V_seq = QKVs
            Q_len, V_len = None, None
        elif len(QKVs) == 5:
            Q_seq, K_seq, V_seq, Q_len, V_len = QKVs
        else:
            raise ValueError("QKVs must be a list of 3 or 5 tensors.")

        if not hasattr(self, 'WQ'):
            self.build([Q_seq.shape, K_seq.shape, V_seq.shape])

        # Linear projections
        Q = self.WQ(Q_seq)  # (batch_size, Q_seq_len, output_dim)
        K = self.WK(K_seq)  # (batch_size, K_seq_len, output_dim)
        V = self.WV(V_seq)  # (batch_size, V_seq_len, output_dim)

        # Reshape for multi-head attention
        batch_size = Q.size(0)

        Q = Q.view(batch_size, -1, self.multiheads, self.head_dim).transpose(1, 2)  # (batch_size, multiheads, Q_seq_len, head_dim)
        K = K.view(batch_size, -1, self.multiheads, self.head_dim).transpose(1, 2)  # (batch_size, multiheads, K_seq_len, head_dim)
        V = V.view(batch_size, -1, self.multiheads, self.head_dim).transpose(1, 2)  # (batch_size, multiheads, V_seq_len, head_dim)

        # Scaled Dot-Product Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))  # (batch_size, multiheads, Q_seq_len, K_seq_len)

        # Apply masking if necessary
        if self.mask_right:
            mask = torch.triu(torch.ones(scores.size(-2), scores.size(-1)), diagonal=1).bool().to(scores.device)  # (Q_seq_len, K_seq_len)
            scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        if V_len is not None:
            # Assuming V_len is a tensor of shape (batch_size, V_seq_len)
            mask = (V_seq.sum(dim=-1) == 0)  # (batch_size, V_seq_len)
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(1), float('-inf'))

        attn = F.softmax(scores, dim=-1)  # (batch_size, multiheads, Q_seq_len, K_seq_len)
        attn = F.dropout(attn, p=0.1, training=self.training)

        output = torch.matmul(attn, V)  # (batch_size, multiheads, Q_seq_len, head_dim)

        # Concatenate heads
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.output_dim)  # (batch_size, Q_seq_len, output_dim)

        if Q_len is not None:
            # Assuming Q_len is a tensor of shape (batch_size, Q_seq_len)
            mask = (Q_seq.sum(dim=-1) == 0)  # (batch_size, Q_seq_len)
            output = output * mask.unsqueeze(-1)

        return output  # (batch_size, Q_seq_len, output_dim)


class ComputeMasking(nn.Module):
    """Compute if inputs contain zero value.

    Returns:
        torch.Tensor: Mask tensor where True indicates non-zero values.
    """

    def __init__(self):
        """Initialization for ComputeMasking."""
        super(ComputeMasking, self).__init__()

    def forward(self, inputs):
        """Compute mask where inputs are not equal to zero.

        Args:
            inputs (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Mask tensor of type float32.
        """
        mask = inputs != 0
        return mask.float()


class OverwriteMasking(nn.Module):
    """Set values at specific positions to zero based on mask.

    Args:
        inputs (list): [value tensor, mask tensor].

    Returns:
        torch.Tensor: Tensor after applying mask.
    """

    def __init__(self):
        """Initialization for OverwriteMasking."""
        super(OverwriteMasking, self).__init__()

    def forward(self, inputs):
        """Apply mask to inputs.

        Args:
            inputs (list of torch.Tensor): [value tensor, mask tensor].

        Returns:
            torch.Tensor: Masked tensor.
        """
        return inputs[0] * inputs[1].unsqueeze(-1)


def PersonalizedAttentivePooling(dim1, dim2, dim3, seed=0):
    """Soft alignment attention implementation.

    Attributes:
        dim1 (int): First dimension of value shape.
        dim2 (int): Second dimension of value shape.
        dim3 (int): Dimension of query.

    Returns:
        nn.Module: PersonalizedAttentivePooling module.
    """

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
            """
            Args:
                vecs_input (torch.Tensor): Tensor of shape (batch_size, dim1, dim2).
                query_input (torch.Tensor): Tensor of shape (batch_size, dim3).

            Returns:
                torch.Tensor: Weighted summary of inputs value of shape (batch_size, dim2).
            """
            user_vecs = self.dropout(vecs_input)  # (batch_size, dim1, dim2)
            user_att = self.activation(self.dense(user_vecs))  # (batch_size, dim1, dim3)
            user_att2 = torch.matmul(user_att, query_input.unsqueeze(-1)).squeeze(-1)  # (batch_size, dim1)
            user_att2 = self.softmax(user_att2)  # (batch_size, dim1)
            user_att2 = user_att2.unsqueeze(1)  # (batch_size, 1, dim1)
            user_vec = torch.bmm(user_att2, user_vecs).squeeze(1)  # (batch_size, dim2)
            return user_vec

    return PersonalizedAttentivePoolingModule(dim1, dim2, dim3, seed)


# Example usage (optional, remove if not needed)
# if __name__ == "__main__":
#     # Example tensors
#     batch_size = 2
#     seq_length = 5
#     input_dim = 10
#     dim = 8
#     multiheads = 4
#     head_dim = 16
#     dim1, dim2, dim3 = 5, 10, 8

#     att_layer = AttLayer2(dim=dim, seed=42)
#     input_tensor = torch.randn(batch_size, seq_length, input_dim)
#     output = att_layer(input_tensor)
#     print("AttLayer2 Output:", output.shape)

#     self_att = SelfAttention(multiheads=multiheads, head_dim=head_dim, seed=42, mask_right=True)
#     Q = torch.randn(batch_size, seq_length, input_dim)
#     K = torch.randn(batch_size, seq_length, input_dim)
#     V = torch.randn(batch_size, seq_length, input_dim)
#     output_self_att = self_att([Q, K, V])
#     print("SelfAttention Output:", output_self_att.shape)

#     compute_mask = ComputeMasking()
#     mask = compute_mask(input_tensor)
#     print("ComputeMasking Output:", mask.shape)

#     overwrite_mask = OverwriteMasking()
#     masked_output = overwrite_mask([input_tensor, mask])
#     print("OverwriteMasking Output:", masked_output.shape)

#     pooling = PersonalizedAttentivePooling(dim1, dim2, dim3, seed=42)
#     query = torch.randn(batch_size, dim3)
#     pooled_output = pooling(input_tensor, query)
#     print("PersonalizedAttentivePooling Output:", pooled_output.shape)
