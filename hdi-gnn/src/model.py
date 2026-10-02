"""Twin-GCN HDI classifier: one shared encoder on both molecules, concat, MLP head."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class Encoder(nn.Module):
    def __init__(self, in_dim, hid=64):
        super().__init__()
        self.c1 = GCNConv(in_dim, hid)
        self.c2 = GCNConv(hid, hid)

    def forward(self, d):
        x = F.relu(self.c1(d.x, d.edge_index))
        x = F.relu(self.c2(x, d.edge_index))
        return global_mean_pool(x, d.batch)


class HDIModel(nn.Module):
    def __init__(self, in_dim, hid=64):
        super().__init__()
        self.enc = Encoder(in_dim, hid)
        self.head = nn.Sequential(
            nn.Linear(2 * hid, hid), nn.ReLU(), nn.Linear(hid, 1)
        )

    def forward(self, a, b):
        za, zb = self.enc(a), self.enc(b)
        return self.head(torch.cat([za, zb], dim=1)).squeeze(-1)
# ponytail: molecular structure only. fuse CYP/transporter + PK features by
# concatenating them into the head input — add when structure alone underperforms.
