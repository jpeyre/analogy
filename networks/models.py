from __future__ import division
from networks.language_layers import LanguageEmb
from networks.criterions import RankingCriterionBidir, BCESoftmax
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from networks.models_base import Net
import torch
import numpy as np

def get_model(opt):
    network = opt.network
    models = {'netindepemb': lambda opt:NetIndepEmb(opt),
              'netindepclassif': lambda opt:NetIndepClassif(opt)}
    return models[network](opt)



class NetIndepClassif(Net):

    def __init__(self, opt):
        super(NetIndepClassif, self).__init__(opt)

        # Compute gram_id to have module <-> gram correspondencies
        self.gram_id = {}
        count = 0
        for gram, is_active in self.activated_grams.iteritems():
            if is_active:
                self.gram_id[gram] = count
                count +=1 

        # In classification set-up: only defined visual nets
        self.visual_nets = nn.ModuleList()
        for gram, is_active in self.activated_grams.iteritems():
            if is_active:

                vis_layer = nn.Sequential(self.get_module_vis(self.modules_name[gram], normalize=False),\
                                        nn.Linear(self.embed_size, len(self.vocab[gram])))
                self.visual_nets.append(vis_layer)


        # Same criterion log-loss
        self.criterions = {}
        for _, gram in enumerate(self.activated_grams):
            if self.activated_grams[gram] and gram not in self.criterions:
                self.criterions[gram] = nn.MultiLabelSoftMarginLoss()


        # Define parameters and optimizer
        self.params = filter(lambda p: p.requires_grad, self.parameters()) 

        if self.optim_name == 'adam':
            self.optimizer = optim.Adam(self.params, lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optim_name == 'sgd':
            self.optimizer = optim.SGD(self.params, lr=self.learning_rate, momentum=self.momentum, weight_decay=self.weight_decay)
        else:
            print('Choose optimizer')


    def forward(self, batch_input):

        scores = {}
        labels = {}

        for _, gram in enumerate(self.activated_grams):
            if self.activated_grams[gram]:

                scores[gram]    = self.visual_nets[self.gram_id[gram]](batch_input)
                _, labels[gram] = self.get_queries(batch_input, self.sample_negatives, gram)

        return (scores, labels)


    def get_scores(self, batch_input):

        scores_grams, _ = self(batch_input)

        for _, gram in enumerate(self.activated_grams):
            if self.activated_grams[gram]:
                scores_grams[gram] = F.sigmoid(self.scale_criterion*scores_grams[gram])

        return scores_grams


    def train_(self, batch_input):

        self.optimizer.zero_grad()

        loss_all = {}
        tp_all = {}
        fp_all = {}
        num_pos_all = {}

        scores_grams,_ = self(batch_input)

        for gram, is_active in self.activated_grams.iteritems():
            if is_active:
                if self.criterions[gram]:

                    scores         = scores_grams[gram]
                    labels         = batch_input['labels_'+gram].type(scores.data.type())
                    loss_all[gram] = self.criterions[gram](scores, labels)
                    activations    = F.sigmoid(scores)
                    tp_all[gram], fp_all[gram], num_pos_all[gram] = self.get_statistics(activations, labels)

        loss = 0
        for _,val in loss_all.iteritems():
            loss += val

        loss.backward()
        self.optimizer.step()

        return loss_all, tp_all, fp_all, num_pos_all


    def val_(self, batch_input):

        loss_all = {}
        tp_all = {}
        fp_all = {}
        num_pos_all = {}

        scores_grams,_ = self(batch_input)

        for gram, is_active in self.activated_grams.iteritems():
            if is_active:
                if self.criterions[gram]:

                    scores         = scores_grams[gram]
                    labels         = batch_input['labels_'+gram].type(scores.data.type())
                    loss_all[gram] = self.criterions[gram](scores, labels)
                    activations    = F.sigmoid(scores) 
                    tp_all[gram], fp_all[gram], num_pos_all[gram] = self.get_statistics(activations, labels)


        return loss_all, tp_all, fp_all, num_pos_all



class NetIndepEmb(Net):

    def __init__(self, opt):
        super(NetIndepEmb, self).__init__(opt)

        # Compute gram_id to have module <-> gram correspondencies
        self.gram_id = {}
        count = 0
        for gram, is_active in self.activated_grams.iteritems():
            if is_active:
                self.gram_id[gram] = count
                count +=1 

        ######################
        """ Visual network """
        ######################

        self.visual_nets = nn.ModuleList()
        for gram, is_active in self.activated_grams.iteritems():
            if is_active:
                vis_layer = self.get_module_vis(self.modules_name[gram], normalize=self.normalize_vis)
                self.visual_nets.append(vis_layer)

        ########################
        """ Language network """
        ########################

        self.language_nets = nn.ModuleList()
        for gram, is_active in self.activated_grams.iteritems():
            if is_active:
                lang_layer = self.get_module_language(  self.net_language_name, \
                                                        num_words=self.num_words[gram], \
                                                        vocab_size=len(self.vocab[gram]), \
                                                        normalize=self.normalize_lang)
                self.language_nets.append(lang_layer)


        #################
        """ Criterion """
        #################

        self.criterions = {}
        for gram, is_active in self.activated_grams.iteritems():
            if is_active:
                if self.criterion_name=='bidirectional_ranking':
                    self.criterions[gram] = RankingCriterionBidir(margin=self.margin)
                elif self.criterion_name=='logloss':
                    self.criterions[gram] = nn.MultiLabelSoftMarginLoss()
                elif self.criterion_name=='bcesoftmax':
                    self.criterions[gram] = BCESoftmax()

        ###############
        """ Analogy """
        ###############

        if self.use_analogy:
            self.reg_network = self.get_gamma(input_dim = self.embed_size)

        #################
        """ Optimizer """
        #################

        # Freezing some parts of the network
        num_parameters = len(list(self.parameters()))
        grams_freeze = []
        if self.use_analogy:
            grams_freeze += ['s','r','o']

        for gram in grams_freeze:
            for param in self.language_nets[self.gram_id[gram]].parameters():
                param.requires_grad = False

            for param in self.visual_nets[self.gram_id[gram]].parameters():
                param.requires_grad = False


        self.params = filter(lambda p: p.requires_grad, self.parameters())

        print('Freezing {} parameters out of {}'.format(num_parameters-len(list(self.params)), num_parameters))

        if self.optim_name == 'adam':
            self.optimizer = optim.Adam(self.params, lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optim_name == 'sgd':
            self.optimizer = optim.SGD(self.params, lr=self.learning_rate, momentum=self.momentum, weight_decay=self.weight_decay)
        else:
            print('Optimizer not recognized')



    def train_(self, batch_input):
        """ Keep own train_ as forward scores into sigmoid to get the activations (because the scores are not in [0,1])"""

        self.optimizer.zero_grad()
        loss_all = {}
        tp_all = {}
        fp_all = {}
        num_pos_all = {}

        # Gram branches w/o analogy
        scores, labels = self(batch_input)
        for gram, is_active in self.activated_grams.iteritems():
            if is_active:
                if self.criterions[gram]:

                    if 'ranking' in self.criterion_name:
                        loss_all[gram] = self.criterions[gram](scores[gram], labels[gram])
                        activations = (scores[gram] + 1) / 2

                    elif self.criterion_name=='logloss':
                        loss_all[gram] = self.criterions[gram](self.scale_criterion*scores[gram], labels[gram].float()) 
                        activations = F.sigmoid(self.scale_criterion*scores[gram])

                    elif self.criterion_name=='bcesoftmax':
                        loss_all[gram] = self.criterions[gram](scores[gram], labels[gram].float())
                        activations = F.softmax(scores[gram])

                    tp_all[gram], fp_all[gram], num_pos_all[gram] = self.get_statistics(activations, labels[gram])


        # Analogy part
        if self.use_analogy:

            # Get the visual features for batch (i.e. break down forward here) and detach them
            vis_feats = self.get_visual_features(batch_input, 'sro') #(batch_size, embed_size)

            if self.detach_vis:
                vis_feats = vis_feats.detach()

            # Get target queries: here, the target queries are the positive triplets -> if there is multilabeling we duplicate the corresponding vis features
            queries, labels = self.form_cand_queries_batch(batch_input, 'sro')


            # We do not queries involving no interaction
            idx_pos     = (queries[:,:,1]!=len(self.vocab['o'])).nonzero()
            if len(idx_pos)>0:
                queries_pos = queries.index_select(1,idx_pos[:,1])
                labels_pos  = labels.index_select(1,idx_pos[:,1])
            else:
                loss_all['reg'] = loss_all['sro']
                tp_all['reg'] = tp_all['sro']
                fp_all['reg'] = fp_all['sro']
                num_pos_all['reg'] = num_pos_all['sro']
                return (loss_all, tp_all, fp_all, num_pos_all)


            # Get the language features by analogy 
            lang_feats_analogy  = self.get_language_features_analogy(queries_pos)

            # Compute similarity
            scores_analogy = self.compute_similarity(vis_feats, lang_feats_analogy, 'sro') 
            activations    = F.sigmoid(self.scale_criterion*scores_analogy)

            # Loss adds up
            loss_all['reg'] = self.criterions['sro'](self.scale_criterion*scores_analogy, labels_pos.float()) # Rescale before sigmoid (vanishing gradients)

            # Statistics
            tp_all['reg'], fp_all['reg'], num_pos_all['reg'] = self.get_statistics(activations, labels_pos)


        # Combine losses
        if self.use_analogy:
            loss = loss_all['sro'] + self.lambda_reg*loss_all['reg']
        else:
            loss = 0
            for _, val in loss_all.iteritems():
                loss += val

        # Gradient step
        loss.backward() 
        self.optimizer.step() 


        # Update the embeddings of source visual phrase 
        if self.use_analogy and self.precomp_vp_source_embedding:
            self.eval()
            for gram in self.queries_source.keys():
                lang_feats_precomp_source_gram       = self.get_language_features(self.queries_source[gram], 'sro')
                self.lang_feats_precomp_source[gram] = lang_feats_precomp_source_gram.detach()
            self.train()


        return (loss_all, tp_all, fp_all, num_pos_all)


    def val_(self, batch_input):
        loss_all = {}
        tp_all = {}
        fp_all = {}
        num_pos_all = {}

        # Gram w/o analogy
        scores, labels = self(batch_input)
        for gram, is_active in self.activated_grams.iteritems():
            if is_active:
                if self.criterions[gram]:

                    if 'ranking' in self.criterion_name:
                        loss_all[gram] = self.criterions[gram](scores[gram], labels[gram])
                        activations = (scores[gram] + 1) / 2

                    elif self.criterion_name=='logloss':
                        loss_all[gram] = self.criterions[gram](self.scale_criterion*scores[gram], labels[gram].float())
                        activations = F.sigmoid(self.scale_criterion*scores[gram])

                    elif self.criterion_name == 'bcesoftmax':
                        loss_all[gram] = self.criterions[gram](scores[gram], labels[gram].float()).data[0]
                        activations = F.softmax(scores[gram])

                    tp_all[gram], fp_all[gram], num_pos_all[gram] = self.get_statistics(activations, labels[gram])


        # Analogy part
        if self.use_analogy:

            # Get the visual features for batch (i.e. break down forward here) and detach them (we do not back-prop on visual features)
            vis_feats       = self.get_visual_features(batch_input, 'sro').detach()
            queries, labels = self.form_cand_queries_batch(batch_input, 'sro')

            # For analogy transformation: remove queries involving no interaction
            idx_pos     = (queries[:,:,1]!=len(self.vocab['o'])).nonzero()
            queries_pos = queries.index_select(1,idx_pos[:,1])
            labels_pos  = labels.index_select(1,idx_pos[:,1])

            # Get the language features by analogy 
            lang_feats_analogy  = self.get_language_features_analogy(queries_pos)

            # Compute similarity
            scores_analogy = self.compute_similarity(vis_feats, lang_feats_analogy, 'sro')
            activations    = F.sigmoid(self.scale_criterion*scores_analogy)

            # Loss adds up
            loss_all['reg'] = self.criterions['sro'](self.scale_criterion*scores_analogy, labels_pos.float()) # Rescale before sigmoid (vanishing gradients)

            # Get statistics
            tp_all['reg'], fp_all['reg'], num_pos_all['reg'] = self.get_statistics(activations, labels_pos)


        return (loss_all, tp_all, fp_all, num_pos_all)


    def get_scores(self, batch_input):
        """ Scores produced by independent branches: p(s), p(o), p(r), p(sr), p(ro), p(sro) """

        scores_grams, _ = self(batch_input)

        for _, gram in enumerate(self.activated_grams):
            if self.activated_grams[gram]:
                
                if 'ranking' in self.criterion_name:
                    scores_grams[gram] = (scores_grams[gram] + 1) / 2

                elif self.criterion_name=='logloss':
                    scores_grams[gram] = F.sigmoid(self.scale_criterion*scores_grams[gram])

                elif self.criterion_name=='bcesoftmax':
                    scores_grams[gram] = F.softmax(scores_grams[gram])

        return scores_grams



