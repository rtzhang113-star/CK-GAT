import torch.nn as nn
import torch.nn.functional as F
import dgl.nn as dglnn


class DGLGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, heads, drop):
        super().__init__()
        self.gat = dglnn.GATConv(
            in_feats=in_dim,
            out_feats=out_dim,
            num_heads=heads,
            feat_drop=drop,
            attn_drop=drop,
            residual=True,
            activation=F.silu,
        )

        total_out_dim = out_dim * heads
        self.ln = nn.LayerNorm(total_out_dim)

    def forward(self, g, h):
        h_out = self.gat(g, h)

        h_out_flattened = h_out.flatten(1)

        return self.ln(h_out_flattened)
