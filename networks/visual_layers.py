from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.misc_layers import Gated_Embedding_Unit, Proj_2Unit, Proj_1Unit



class VisualPrecomp(nn.Module):
    """
    Visual module with pre-extracted visual features: simple projection fc layer + normalization
    """
    def __init__(self, embed_size, d_appearance=1024, gated_unit=False, normalize=False, num_layers=2):
        super(VisualPrecomp, self).__init__()

        self.embed_size = embed_size
        self.d_appearance = d_appearance
        self.num_layers = num_layers
        self.normalize = normalize

        # Projection layer
        if gated_unit:
            self.proj = Gated_Embedding_Unit(d_appearance, self.embed_size)
        else:
            if self.num_layers==2:
                self.proj = Proj_2Unit(d_appearance, self.embed_size, normalize=normalize, dropout=True)
            elif self.num_layers==1:
                self.proj = Proj_1Unit(d_appearance, self.embed_size, normalize=normalize)

    def forward(self, precomp_feats):

        if self.num_layers>0:
            feats = self.proj(precomp_feats)
        else:
            if self.normalize:
                feats = F.normalize(precomp_feats,2,-1) # L2 norm according to last dimension
            else:
                feats = precomp_feats

        return feats



