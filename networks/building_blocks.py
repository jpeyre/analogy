from __future__ import division
import torch.nn as nn
from spatial_layers import CroppedBoxCoordinates
from misc_layers import Proj_1Unit, Proj_2Unit, Gated_Embedding_Unit 
import torch
import torch.nn.functional as F


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



class AppearancePrecomp(nn.Module):
    def __init__(self, embed_size, name, d_appearance=1024, d_hidden=1024, normalize=False, dropout=True, num_layers=2):
        super(AppearancePrecomp, self).__init__()

        self.name = name
        self.num_layers = num_layers
        self.normalize = normalize

        # Projection layer
        if self.num_layers==2:
            self.proj = Proj_2Unit(d_appearance, embed_size, hidden_dimension=d_hidden, normalize=normalize, dropout=dropout)
        elif self.num_layers==1:
            self.proj = Proj_1Unit(d_appearance, embed_size, normalize=normalize)


    def forward(self, batch_input):

        if self.name=='subject':
            output = batch_input['precompappearance'][:,0,:]
        elif self.name=='object':
            output = batch_input['precompappearance'][:,1,:]

        if self.num_layers>0:
            output = self.proj(output) # normalization is included
        else:
            if self.normalize:
                output = F.normalize(output,2,-1) # L2 norm according to last dimension

        return output



class SpatialAppearancePrecomp(nn.Module):
    def __init__(self, embed_size, d_appearance=1024, d_hidden=1024, normalize=False, dropout=True, num_layers=2):
        super(SpatialAppearancePrecomp, self).__init__()

        self.normalize = normalize
        self.num_layers = num_layers

        # 2 fc layer + normalize
        self.spatial_module = SpatialRawCrop(400, normalize=True)

        # Project appearance feats in subspace 300 (mimic PCA iccv17) before concatenating with spatial
        self.appearance_module = nn.Linear(d_appearance,300)

        # Aggregate spatial and appearance feature with fc layer
        if self.num_layers==2:
            self.proj = Proj_2Unit(400+600, embed_size, hidden_dimension=d_hidden, normalize=normalize, dropout=dropout)
        elif self.num_layers==1:
            self.proj = Proj_1Unit(400+600, embed_size, normalize=normalize)


    def forward(self, batch_input):

        # Spatial feats
        spatial_feats = self.spatial_module(batch_input) # already L2 norm

        # Appearance feats subject L2 norm 
        appearance_human = batch_input['precompappearance'][:,0,:]
        appearance_human = F.normalize(self.appearance_module(appearance_human))

        # Appearance feats object L2 norm
        appearance_object = batch_input['precompappearance'][:,1,:]
        appearance_object = F.normalize(self.appearance_module(appearance_object))

        # Concat both L2 norm
        appearance_feats = torch.cat([appearance_human, appearance_object],1)
        appearance_feats = F.normalize(appearance_feats)

        # Concat appearance and spatial
        output = torch.cat([spatial_feats, appearance_feats],1)

        # Proj
        if self.num_layers > 0:
            output = self.proj(output)
     
        else: 
            if self.normalize:
                output = F.normalize(output,2,-1) # L2 norm according to last dimension
                #output = F.normalize(output) #old code: check this was doing the same

        return output



class SpatialRawCrop(nn.Module):
    """
    Baseline model using only spatial coordinates of subject and object boxes, i.e renormalized [x1, y1, w1, h1, x2, y2, w2, h2] in the coordinates of union boxes
    """
    def __init__(self, embed_size, normalize=False):
        super(SpatialRawCrop, self).__init__()

        self.embed_size = embed_size
        self.normalize = normalize

        self.raw_coordinates = CroppedBoxCoordinates()

        self.net = nn.Sequential(nn.Linear(8, 128),
                                nn.ReLU(),
                                nn.Linear(128,self.embed_size),
                                nn.ReLU())


    def forward(self, batch_input):

        pair_objects = batch_input['pair_objects']
        output = self.raw_coordinates(pair_objects)
        output = self.net(output)

        if self.normalize:
            output = F.normalize(output)

        return output



