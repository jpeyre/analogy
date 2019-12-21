import torch.nn as nn
import torch.nn.functional as F

class View(nn.Module):
    def __init__(self, outdim):
        super(View, self).__init__()
        self.outdim = outdim

    def forward(self, x):
        return x.view(-1, self.outdim)


class Proj_2Unit(nn.Module):
    def __init__(self, input_dimension, output_dimension, hidden_dimension=None, normalize=False, dropout=True):
        super(Proj_2Unit, self).__init__()

        if not hidden_dimension:
            hidden_dimension = input_dimension
        self.normalize = normalize

        if dropout:
            self.proj = nn.Sequential(nn.Linear(input_dimension, hidden_dimension),
                                    nn.ReLU(),
                                    nn.Dropout(),
                                    nn.Linear(hidden_dimension, output_dimension))

        else:

            self.proj = nn.Sequential(nn.Linear(input_dimension, hidden_dimension),
                                    nn.ReLU(),
                                    nn.Linear(hidden_dimension, output_dimension))


    def forward(self,x):

        x = self.proj(x)

        if self.normalize:
            x = F.normalize(x,2,-1) # L2 norm according to last dimension

        return x


class Proj_1Unit(nn.Module):
    def __init__(self, input_dimension, output_dimension, normalize=False):
        super(Proj_1Unit, self).__init__()

        self.normalize = normalize
        self.proj = nn.Linear(input_dimension, output_dimension)

    def forward(self,x):

        x = self.proj(x)

        if self.normalize:
            x = F.normalize(x,2,-1) # L2 norm according to last dimension

        return x


class Gated_Embedding_Unit(nn.Module):
    def __init__(self, input_dimension, output_dimension):
        super(Gated_Embedding_Unit, self).__init__()

        self.fc = nn.Linear(input_dimension, output_dimension)
        self.cg = Context_Gating(output_dimension)
  
    def forward(self,x):
        
        x = self.fc(x)
        x = self.cg(x)
        x = F.normalize(x,2,-1)

        return x


class Context_Gating(nn.Module):
    """ Taken from Miech et al. """
    def __init__(self, dimension, add_batch_norm=True):
        super(Context_Gating, self).__init__()
        self.fc = nn.Linear(dimension, dimension)
        self.add_batch_norm = add_batch_norm
        self.batch_norm = nn.BatchNorm1d(dimension)
        
    def forward(self,x):
        x1 = self.fc(x)

        if self.add_batch_norm:
            x1 = self.batch_norm(x1) 

        x = torch.cat((x, x1), 1)
        
        return F.glu(x,1)


