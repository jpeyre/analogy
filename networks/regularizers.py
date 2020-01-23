"""
Library analogy transformations
"""
import torch.nn as nn
import torch
import torch.nn.functional as F


class RegAdditive(nn.Module):
    def __init__(self, embed_size, input_dim=900, use_proj=False):
        super(RegAdditive, self).__init__()

        if not input_dim:
            input_dim = 3*embed_size

        self.use_proj = use_proj

        if self.use_proj:
            self.proj = nn.Linear(input_dim, embed_size, bias=False)

    def forward(self, quadruplets_emb):

        deformation = torch.cat((quadruplets_emb['target_s'] - quadruplets_emb['source_s'],\
                                quadruplets_emb['target_r'] - quadruplets_emb['source_r'],\
                                quadruplets_emb['target_o'] - quadruplets_emb['source_o']),1)

        if self.use_proj:
            deformation = self.proj(deformation)

        return deformation


class RegDeep(nn.Module):
    def __init__(self, embed_size, input_dim=900):
        super(RegDeep, self).__init__()

        if not input_dim:
            input_dim=3*embed_size

        self.A = nn.Sequential(nn.Linear(input_dim,512, bias=False),\
                                        nn.ReLU(),\
                                        nn.Linear(512,embed_size, bias=False))

    def forward(self, quadruplets_emb):

        deformation = torch.cat((quadruplets_emb['target_s'] - quadruplets_emb['source_s'],\
                                quadruplets_emb['target_r'] - quadruplets_emb['source_r'],\
                                quadruplets_emb['target_o'] - quadruplets_emb['source_o']),1)

        deformation = self.A(deformation)

        return deformation


class RegDeepCond(nn.Module):
    def __init__(self, embed_size, input_dim=900):
        super(RegDeepCond, self).__init__()

        if not input_dim:
            input_dim=3*embed_size

        self.A = nn.Sequential(nn.Linear(2*embed_size,512, bias=False),\
                                        nn.ReLU(),\
                                        nn.Linear(512,embed_size, bias=False))

        self.proj = nn.Linear(input_dim, embed_size, bias=False)

    def forward(self, quadruplets_emb):

        deformation = torch.cat((quadruplets_emb['target_s'] - quadruplets_emb['source_s'],\
                                quadruplets_emb['target_r'] - quadruplets_emb['source_r'],\
                                quadruplets_emb['target_o'] - quadruplets_emb['source_o']),1)

        deformation = self.proj(deformation)

        deformation = self.A(torch.cat((quadruplets_emb['source_sro'], deformation),1))

        return deformation


