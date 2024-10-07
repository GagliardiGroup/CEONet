import torch
import torch.nn as nn
import torch.nn.functional as F

class DistHead(nn.Module):
    #One head of self-attention
    def __init__(self,n_emb,head_size,cutoff=4.0):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(n_emb, head_size, bias=False)
        self.query = nn.Linear(n_emb, head_size, bias=False)
        self.value = nn.Linear(n_emb, head_size, bias=False)
        #learnable exponential weight:
        self.invr0 = nn.Parameter((1.0 / cutoff) * (torch.rand(1) + 0.5))
        # self.dropout = nn.Dropout(dropout)

    def forward(self,X,Z,ptr):
        #X -- features
        #Z -- positions
        K = self.key(X)
        Q = self.query(X)
        V = self.value(X)

        #Attention mechanism
        wei = Q @ K.T * self.head_size**-0.5
        
        #Only attention between els in the same molecule
        blockdiag = torch.block_diag(*[torch.ones(n,n) for n in torch.diff(ptr)]).to(X.device)
        wei = wei.masked_fill(blockdiag==0,float('-inf'))
        wei = F.softmax(wei,dim=-1)
        
        #Multiply by exponential distance decay
        expd = torch.exp(-1.0 * self.invr0 * torch.cdist(Z,Z))
        wei = wei * expd

        #Regularization
        # wei = self.dropout(wei)

        #Multiply by values
        out = wei @ V
        return out

class MultiHeadAttention(nn.Module):
    #Multiple heads of self-attention
    def __init__(self,n_emb,head_size,num_heads,cutoff=4.0):
        super().__init__()
        self.heads = nn.ModuleList([DistHead(n_emb,head_size,cutoff=cutoff) for _ in range(num_heads)])
        self.proj = nn.Linear(int(head_size*num_heads),n_emb)
        # self.dropout = nn.Dropout(dropout)

    def forward(self,X,Z,ptr):
        #Concatenate over channel dimension
        out = torch.cat([h(X,Z,ptr) for h in self.heads], dim=-1)
        out = self.proj(out)
        # out = self.dropout(out)
        return out

class FeedForward(nn.Module):
    def __init__(self,n_emb):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_emb,4*n_emb),
            nn.ReLU(),
            nn.Linear(4*n_emb,n_emb),
            # nn.Dropout(dropout),
        )

    def forward(self,x):
        return self.net(x)

class Block(nn.Module):
    #communication + computation
    def __init__(self, n_emb, n_head, cutoff=4.0):
        super().__init__()
        head_size = n_emb // n_head #conserve total size
        self.sa = MultiHeadAttention(n_emb,head_size,n_head,cutoff=cutoff)
        self.ffwd = FeedForward(n_emb)
        # self.ln1 = nn.LayerNorm(n_emb)
        # self.ln2 = nn.LayerNorm(n_emb)

    def forward(self,X,Z,ptr):
        # X = X + self.sa(self.ln1(X))
        # X = X + self.ffwd(self.ln2(X))
        X = X + self.sa(X,Z,ptr)
        X = X + self.ffwd(X)
        return X