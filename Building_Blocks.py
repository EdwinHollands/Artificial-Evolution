import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# MULTI-HEAD ATTENTION -------------------------------------
class MultiHead(nn.Module):
    def __init__(self, Q_dim, heads, dropout=0, cross=False):
        super().__init__()
        self.layer_dim = head_dim*heads
        self.heads = heads
        self.Q_dim = Q_dim
        self.query = nn.Linear(layer_dim, layer_dim, bias=False) #questions
        self.key = nn.Linear(layer_dim, layer_dim, bias=False) #detecting relevance
        self.value = nn.Linear(layer_dim, layer_dim, bias=False) #answers
        self.project = nn.Linear(layer_dim, layer_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        B,T,C = inputs.shape
        heads = self.heads
        Q_dim = self.Q_dim
        q = self.query(inputs)
        k = self.key(inputs)
        v = self.value(inputs)
        q = q.view(B, T, heads, Q_dim).transpose(1, 2)
        k = k.view(B, T, heads, Q_dim).transpose(1, 2)
        v = v.view(B, T, heads, Q_dim).transpose(1, 2)
        Aff = q @ k.transpose(-2,-1) * Q_dim **-0.5
        Aff = F.softmax(Aff, dim=-1) # softmax: exp then normalise along rows
        Aff = self.dropout(Aff)
        out = Aff @ v # multiply affinities with values (B, n_heads, T, T) @ (B, n_heads, T, head_size) → (B, n_heads, T, head_size)
        out = out.transpose(1, 2).contiguous().view(B, T, C) #reassembles
        return self.dropout(self.project(out))

# MULTI-LAYER PEREPTRON --------------------------------------
class MLP(nn.Module):
    def __init__(self, layer_dims):
        super().__init__()
        self.layers = len(layer_dims)
        self.layer_dims = layer_dims
        self.linears = nn.ModuleList([
            nn.Linear(layer_dims[i], layer_dims[i+1]) 
            for i in range(len(layer_dims)-1)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(dim)
            for dim in layer_dims[1:-1]
        ])
        #normalise hidden layers

    def forward(self, inputs):
        out = inputs
        for i, linear in enumerate(self.linears[:-1]):
            out = linear(out)
            out = self.norms[i](out)
            out = F.relu(out)
        return self.linears[-1](out)

# LAYER OF ATTENTION AND PERCEPTRON
class Layer(nn.Module):
    def __init__(self, layer_dim):
        super().__init__()
        self.sa = MultiHead()
        self.ffwd = FeedForward(layer_dim, ff_scalar)
        self.ln1  = nn.LayerNorm(layer_dim) #layer norms fix means and deviations for each token then does a linear transform
        self.ln2  = nn.LayerNorm(layer_dim)

    def forward(self, inputs):
        inputs = inputs + self.sa(self.ln1(inputs)) #residuals are important for deep networks!
        inputs = inputs + self.ffwd(self.ln2(inputs))
        return inputs