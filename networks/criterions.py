from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Contains customized loss functions
"""

class BCESoftmax(nn.Module):
    def __init__(self):
        super(BCESoftmax, self).__init__()

    def forward(self, scores, target):

        loss = F.binary_cross_entropy(F.softmax(scores),target) 

        return loss


class MseLoss(nn.Module):
    def __init__(self):
        super(MseLoss, self).__init__()

    def mse_loss(input, target):
        return torch.sum((input - target)**2) / input.data.nelement()


class MultiLabelPositiveLoss(nn.Module):
    """
    MultiLabel Loss that only back-propagate on positive examples
    loss(x,y) = - sum_i sum_c y_ic log(1/(1+exp(-xi)))
    """

    def __init__(self):
        super(MultiLabelPositiveLoss, self).__init__()

    def forward(self, scores, target):
        """
        scores: (N,K) after sigmoid
        target: (N,K) in [0,1]
        """
        scores = torch.log(scores.clamp(min=1e-8)) # add torch.clamp else small values explode in backward !
        loss = -target*scores
        loss = loss.mean()

        return loss



class CustomRankingCriterion(nn.Module):
    """
    TODO: implement variant where you average over the negatives
    """
    def __init__(self, margin=0):
        super(CustomRankingCriterion, self).__init__()
        self.margin = margin


    def forward(self, scores, labels):
        """
        Take as input a matrix of similarity scores of size (N,M) where N corresponds to the visual modality, M to the language modality
        E.g. N would be the number of pairs of boxes, M would be the number of triplets
        You have to specify the labels (N,M) as input (in standard image captioning, it would be diagonal), but in our case where we encounter multilabeling and the vocabulary is fixed on small, this is often non diagonal
        You need to speficify this else, with standard ranking loss, you would penalized good labels 
        """
        inf_num = 10.0

        # Sign of scores
        sign_scores = (scores.clone()>=0).float()
        sign_scores.data[sign_scores.data==0] = -1

        # Get the minimum scores over positives for each pair
        constant_matrix = scores.clone()
        constant_matrix.data.fill_(0)
        constant_matrix[labels.data==0] = 10.0 # the positives are increased by a large number so that when doing the min there is no chance that we select a negative
        pos_scores = scores + constant_matrix
        min_pos = pos_scores.min(1)[0]

        # Get the maximum scores over negatives for each pair
        constant_matrix = scores.clone()
        constant_matrix.data.fill_(0)
        constant_matrix[labels.data==1] = -10.0
        neg_scores = scores + constant_matrix # the negatives are be decreased by a large number (the scores are between -1,1) so that when doing the max, there is no chance that we select a positive
        max_neg = neg_scores.max(1)[0]

        # Minimum score over positives should be above maximum score over negatives
        cost = (self.margin + max_neg - min_pos).clamp(min=0)


        return cost.mean()



class RankingCriterionBidir(nn.Module):
    def __init__(self, margin=0, square=False):
        super(RankingCriterionBidir, self).__init__()
        self.margin = margin
        self.square = square

    def forward(self, scores, labels):
        """
        Instead of doing max(0, m + max_{neg} sim - min_{pos} sim ) for each batch we do: sum_{neg} sum_{pos} max(0, m + sim_neg - sim_pos ) 
        """

        # For loop over batch
        cost = 0
        count = 0

        # Terms <v_k,w_j> - <v_k,w_k>
        for b in range(scores.size(0)):

            scores_pos = scores[b][labels[b]==1] 
            scores_neg = scores[b][labels[b]==0]

            num_pos = len(scores_pos)
            num_neg = len(scores_neg)

            # Some lines will be passed as they do not have labels (for sr,ro,sro, there is no background class, contrary to r which has a no_interaction class)
            if num_pos>0 and num_neg>0:
                
                # Form diff of scores
                scores_pos = scores_pos.unsqueeze(1).expand(num_pos, num_neg)
                scores_neg = scores_neg.unsqueeze(0).expand(num_pos, num_neg)

                scores_diff = (self.margin + scores_neg - scores_pos).clamp(min=0)
                if self.square:
                    scores_diff = scores_diff**2 
                cost += scores_diff.mean()
                count += 1

        # Terms <v_j,w_k> - <v_k,w_k>: involves pairs with no labels as negatives
        for r in range(scores.size(1)):

            scores_pos = scores[:,r][labels[:,r]==1] 
            scores_neg = scores[:,r][labels[:,r]==0]

            num_pos = len(scores_pos)
            num_neg = len(scores_neg)

            if num_pos>0 and num_neg>0:
                
                # Form diff of scores
                scores_pos = scores_pos.unsqueeze(1).expand(num_pos, num_neg)
                scores_neg = scores_neg.unsqueeze(0).expand(num_pos, num_neg)

                scores_diff = (self.margin + scores_neg - scores_pos).clamp(min=0)
                if self.square:
                    scores_diff = scores_diff**2 
                cost += scores_diff.mean()
                count += 1

        cost /= count

        return cost



