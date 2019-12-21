from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, Function
from networks.misc_layers import Gated_Embedding_Unit, Proj_2Unit, Proj_1Unit



class LanguageEmb(nn.Module):
    """
    Embedding module: performs no projection
    For an input of shape (N,num_words) it returns language embeddings (N,num_words,d)
    """
    def __init__(self, word_embeddings, finetune=False):
        super(LanguageEmb, self).__init__()

        self.word_embeddings = word_embeddings
        N_vocab = self.word_embeddings.shape[0]
        d_emb   = self.word_embeddings.shape[1]
        self.emb = nn.Embedding(N_vocab, d_emb)

        # Initialize weights of embedding layer
        self.init_weights()

        # freeze weights of embedding layer if use pre-trained word2vec embedding
        if not finetune:
            self.emb.weight.requires_grad = False


    def forward(self, batch_words):

        embeddings = self.emb(batch_words)

        return embeddings


    def init_weights(self):
        """ Initialize weights of embedding layer with pre-computed w2v vector """
        pretrained_weight = torch.from_numpy(self.word_embeddings)
        self.emb.weight.data.copy_(pretrained_weight)



class LanguageProj(nn.Module):
    """
    Projection of query embeddings (N,num_words,P) into (N,embed_size)
    If preceeding layer is word2vec P=300 and the query embeddings are word2vec features
    If preceeding layer is onehot P=vocab_size and the query embeddings are onehot vectors
    Different aggregation functions
    input_dim is usually 300 (word2vec dim) but it can be also the size of vocabulary (if working with 1-hot encoding) 
    """
    def __init__(self, embed_size, num_words=1, input_dim=300, num_layers=2, hidden_dimension=None, gated_unit=False, normalize=True, dropout=False, aggreg='concatenation'):
        super(LanguageProj, self).__init__()
 
        self.num_words = num_words 
        self.embed_size = embed_size
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.aggreg = aggreg # How to aggregate across word if num_words>1       

        # Compute dimension of input vector given to proj (this depends on how you aggregate the words)
        if self.aggreg=='concatenation':
            input_dim_words = self.input_dim*self.num_words
        elif self.aggreg=='average':
            input_dim_words = self.input_dim
 
        # Projection layer
        if gated_unit:
            self.proj = Gated_Embedding_Unit(input_dim*self.num_words, self.embed_size) # already normalized by default
        else:
            if num_layers==2:
                if not hidden_dimension:
                    hidden_dimension=input_dim
                self.proj = Proj_2Unit(input_dim_words, self.embed_size, hidden_dimension, normalize, dropout)
            elif num_layers==1:
                # Only 1 layer
                self.proj = Proj_1Unit(input_dim_words, self.embed_size, normalize)


    def forward(self, batch_embeddings):

        if self.aggreg=='concatenation':
            feats = torch.cat([batch_embeddings[:,j,:] for j in range(self.num_words)],1) # Concatenate embeddings to form Nx(300*num_words)
        elif self.aggreg=='average':
            feats = torch.mean(batch_embeddings,1)

        if self.num_layers>0:
            feats = self.proj(feats)

        return feats



