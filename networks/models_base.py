from __future__ import division
from networks.language_layers import LanguageEmb, LanguageProj
import networks.building_blocks as bb
from networks.regularizers import RegAdditive, RegDeep, RegDeepCond
import torch
import torch.nn as nn
import torch.optim as optim
import os.path as osp
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pickle
from torch.nn import Parameter

class Net(nn.Module):

    def __init__(self, opt):
        super(Net, self).__init__()

        self.embed_size           = opt.embed_size
        self.d_hidden             = opt.d_hidden
        self.num_layers           = opt.num_layers
        self.network              = opt.network
        self.word_embeddings      = opt.word_embeddings
        self.sample_negatives     = opt.sample_negatives
        self.d_appearance         = opt.d_appearance
        self.occurrences          = opt.occurrences
        self.start_epoch          = opt.start_epoch
        self.epoch_model          = opt.epoch_model
        self.test_split           = opt.test_split 
        self.cand_test            = opt.cand_test
        self.margin               = opt.margin
        self.use_gpu              = opt.use_gpu
        self.pretrained_model     = opt.pretrained_model
        self.additional_neg_batch = opt.additional_neg_batch
        self.classes              = opt.classes
        self.predicates           = opt.predicates
        self.vocab                = opt.vocab_grams

        # Activated inputs
        self.use_image              = opt.use_image
        self.use_precompappearance  = opt.use_precompappearance
        self.use_precompobjectscore = opt.use_precompobjectscore

        # Optim parameters
        self.learning_rate = opt.learning_rate
        self.weight_decay  = opt.weight_decay
        self.momentum      = opt.momentum
        self.optim_name    = opt.optimizer

        # Analogy parameters
        self.lambda_reg       = opt.lambda_reg
        self.normalize_vis    = opt.normalize_vis
        self.normalize_lang   = opt.normalize_lang
        self.scale_criterion  = opt.scale_criterion
        self.use_analogy      = opt.use_analogy
        self.gamma            = opt.gamma
        self.sim_method       = opt.sim_method
        self.thresh_method    = opt.thresh_method
        self.alpha_r          = opt.alpha_r
        self.alpha_s          = opt.alpha_s
        assert self.alpha_r + self.alpha_s <= 1.0, 'Choose alpha weights below 1'
        self.minimal_mass     = opt.minimal_mass
        self.use_target       = opt.use_target
        self.normalize_source = opt.normalize_source

        self.apply_deformation         = opt.apply_deformation
        self.unique_source_random      = opt.unique_source_random
        self.detach_target             = opt.detach_target
        self.num_source_words_common   = opt.num_source_words_common
        self.restrict_source_subject   = opt.restrict_source_subject
        self.restrict_source_object    = opt.restrict_source_object
        self.restrict_source_predicate = opt.restrict_source_predicate

        # Nets
        self.modules_name = {}
        self.modules_name['s']   = opt.net_unigram_s
        self.modules_name['o']   = opt.net_unigram_o
        self.modules_name['r']   = opt.net_unigram_r
        self.modules_name['sr']  = opt.net_bigram_sr
        self.modules_name['ro']  = opt.net_bigram_ro
        self.modules_name['sro'] = opt.net_trigram_sro
        self.net_language_name   = opt.net_language
        self.criterion_name      = opt.criterion_name
        self.mixture_keys        = opt.mixture_keys.split('_')

        self.precomp_vp_source_embedding = opt.precomp_vp_source_embedding

        self.get_activated_grams()

        self.num_words = {'s': 1,
                          'o': 1,
                          'r': 1,
                          'sr': 2,
                          'ro': 2,
                          'sro': 3}

        # Triplets in vocab
        self.triplets = self.get_triplets_visualphrase() # n_triplet x 3
        self.triplets = Variable(self.triplets)

        self.ite = 0


        """ For speed-up """

        self.idx_sro_to = {}
        self.idx_sro_to_numpy = {}
        for key in opt.idx_sro_to.keys():
            self.idx_sro_to_numpy[key] = opt.idx_sro_to[key].astype(int)
            self.idx_sro_to[key]       = Variable(torch.from_numpy(opt.idx_sro_to[key])).long()
            if self.use_gpu:
                self.idx_sro_to[key] = self.idx_sro_to[key].cuda()

        self.idx_to_vocab = {}
        self.idx_to_vocab_numpy = {}
        for key in opt.idx_to_vocab.keys():
            self.idx_to_vocab_numpy[key] = opt.idx_to_vocab[key].astype(int)
            self.idx_to_vocab[key]       = Variable(torch.from_numpy(opt.idx_to_vocab[key])).long()
            if self.use_gpu:
                self.idx_to_vocab[key] = self.idx_to_vocab[key].cuda()


        self.sim_precomp = None
        self.language_features_precomp = {} 
        self.precomp_triplet_mass()
        self.queries_source = {} 
        self.lang_feats_precomp_source = {} 
        self.triplet_cat_source_common = []


        # Source triplet candidates
        if self.use_analogy:
            self.triplet_cat_source_common = self.precomp_triplet_cat_source(idx_in_vocab_all=True, minimal_mass=self.minimal_mass) 


    def precomp_source_queries(self):
        """ Precompute source queries features """
        self.eval() 
        queries_source = {}

        queries_source['sro'] = Variable(torch.from_numpy(self.triplet_cat_source_common))
        queries_source['s']   = queries_source['sro'].clone()
        queries_source['s'][:,1:] = 0
        queries_source['r']   = queries_source['sro'].clone()
        queries_source['r'][:,[0,2]] = 0
        queries_source['o']   = queries_source['sro'].clone()
        queries_source['o'][:,:2] = 0

        if torch.cuda.is_available():
            for gram in queries_source.keys():
                queries_source[gram] = queries_source[gram].cuda()

        for gram in queries_source.keys():
            lang_feats_precomp_source_gram         = self.get_language_features(queries_source[gram], 'sro')
            self.lang_feats_precomp_source[gram]   = lang_feats_precomp_source_gram.detach()
            self.queries_source[gram]              = queries_source[gram]


    def precomp_target_queries(self, triplet_queries):
        """ Precompute target queries indices """
        self.eval()

        lang_feats_precomp_r = self.get_lang_precomp_feats('r')
        lang_feats_precomp_s = self.get_lang_precomp_feats('s')
        lang_feats_precomp_o = self.get_lang_precomp_feats('o') 

        triplet_queries_idx = np.zeros((len(triplet_queries),3), dtype=np.int)
        queries_sro = Variable(torch.zeros(len(triplet_queries),3).type(lang_feats_precomp_r.data.type())).long()

        for count,triplet_query in enumerate(triplet_queries):

            subjectname, predicate, objectname = triplet_query.split('-')
            sub_cat = self.classes.word2idx[subjectname]
            obj_cat = self.classes.word2idx[objectname]
            rel_cat = self.predicates.word2idx[predicate]

            triplet_queries_idx[count,0] = sub_cat
            triplet_queries_idx[count,1] = rel_cat
            triplet_queries_idx[count,2] = obj_cat

            queries_sro[count,0] = self.idx_to_vocab['s'][sub_cat]
            queries_sro[count,2] = self.idx_to_vocab['o'][obj_cat]
            queries_sro[count,1] = self.idx_to_vocab['r'][rel_cat]

        if torch.cuda.is_available():
            queries_sro = queries_sro.cuda() 


        return queries_sro, triplet_queries_idx


    def precomp_language_features(self):
        """ Precompute language features """
        self.eval()
        self.language_features_precomp['r'] = self.get_lang_precomp_feats('r').data.cpu().numpy()
        self.language_features_precomp['s'] = self.get_lang_precomp_feats('s').data.cpu().numpy()
        self.language_features_precomp['o'] = self.get_lang_precomp_feats('o').data.cpu().numpy()


    def precomp_triplet_cat_source(self, idx_in_vocab_all=False, remove_bg=True, minimal_mass=0):
        """ 
        Precompute candidate source triplets for speed-up 
        Output as [sub_cat, rel_cat, obj_cat] 
        """
        idx_source = np.arange(len(self.vocab['sro'])) # potentially all source triplets in vocab could work

        if remove_bg:
            idx_filter = np.where(self.idx_sro_to_numpy['r'][idx_source]>0)[0] # remove those involving no_interaction (by default)
            idx_source = idx_source[idx_filter]

        if minimal_mass:
            # Check that occurrences are well loaded
            assert len(self.occurrences)>0, 'Occurrences not well loaded'
            idx_filter = np.where(self.triplet_mass[idx_source] > minimal_mass)[0]
            idx_source = idx_source[idx_filter]

        sub_cat_source = self.idx_sro_to_numpy['s'][idx_source]
        obj_cat_source = self.idx_sro_to_numpy['o'][idx_source]
        rel_cat_source = self.idx_sro_to_numpy['r'][idx_source]

        if idx_in_vocab_all:
            sub_cat_source = self.idx_to_vocab_numpy['s'][sub_cat_source]
            obj_cat_source = self.idx_to_vocab_numpy['o'][obj_cat_source]
            rel_cat_source = self.idx_to_vocab_numpy['r'][rel_cat_source] 

            triplet_cat_source = np.hstack((sub_cat_source, rel_cat_source, obj_cat_source))

        else:
            triplet_cat_source = np.vstack((sub_cat_source, rel_cat_source, obj_cat_source)).T

        return triplet_cat_source


    def precomp_triplet_mass(self):
        """ Precompute mass for all triplets in vp vocab """
        self.triplet_mass = np.zeros((len(self.vocab['sro']),))
        for count, triplet in enumerate(self.vocab['sro'].words()):
            self.triplet_mass[count] = self.get_triplet_mass(triplet)


    def get_triplet_mass(self, tripletname):
        """ Determine the weights of a triplet: here, we just count occurrences in training """
        if tripletname not in self.occurrences:
            return 0
        else:
            triplet_mass = self.occurrences[tripletname]

        return triplet_mass



    def precomp_sim_tables(self):
        """ 
        Precomputed similarity tables for s,r,o for all semantic pairs in vocab 
        Considerable speed-up, providing the similarity tables don't evolve (i.e. no finetune)
        """
        self.sim_precomp = {}

        if self.sim_method == 'emb_jointspace':

            for gram in ['s','r','o']:

                vocab_gram = self.vocab[gram]
                V  = len(vocab_gram)
                self.sim_precomp[gram] = np.zeros((V,V))

                for v_target in range(V):

                    word_target         = vocab_gram.idx2word[v_target]
                    query_target        = self.queries_unigrams(word_target, gram)
                    embedding_target    = self.language_nets[self.gram_id[gram]](query_target)


                    for v_source in range(V):

                        word_source         = vocab_gram.idx2word[v_source]
                        query_source        = self.queries_unigrams(word_source, gram)
                        embedding_source    = self.language_nets[self.gram_id[gram]](query_source)

                        self.sim_precomp[gram][v_target, v_source] = torch.mul(embedding_target, embedding_source).sum().data[0]


        elif self.sim_method == 'emb_word2vec':

            for gram in ['s','r','o']:

                vocab_gram = self.vocab[gram]
                V  = len(vocab_gram)
                self.sim_precomp[gram] = np.zeros((V,V))

                for v_target in range(V):

                    embedding_target = self.word_embeddings[self.idx_to_vocab[gram][v_target].data[0]]
                    embedding_target = embedding_target / np.linalg.norm(embedding_target)

                    for v_source in range(V):

                        embedding_source = self.word_embeddings[self.idx_to_vocab[gram][v_source].data[0]]
                        embedding_source = embedding_source / np.linalg.norm(embedding_source)

                        self.sim_precomp[gram][v_target, v_source] = np.sum(embedding_target*embedding_source)


    def get_activated_grams(self):
        self.activated_grams = {'s': 0,
         'o': 0,
         'r': 0,
         'sr': 0,
         'ro': 0,
         'sro': 0}
        for gram in self.activated_grams.keys():
            if self.modules_name[gram]:
                self.activated_grams[gram] = 1

        if np.array(self.activated_grams.values()).sum() == 0:
            print 'Attention we need to activated at least 1 branch of the network'

    def adjust_learning_rate(self, opt, epoch):
        """Sets the learning rate to the initial LR
        decayed by 10 every 30 epochs"""
        lr = opt.learning_rate * 0.1 ** (epoch // opt.lr_update)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def get_optimizer(self):
        if self.optim_name == 'adam':
            self.optimizer = optim.Adam(self.params, lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optim_name == 'sgd':
            self.optimizer = optim.SGD(self.params, lr=self.learning_rate, momentum=self.momentum, weight_decay=self.weight_decay)
        else:
            print 'Choose optimizer'


    def get_statistics(self, scores, labels):
        """
        Expects scores renormalized [0,1]
        """
        activations = (scores > 0.5).float()
        num_pos = labels.data.sum(0).squeeze().cpu()
        tp = (activations * labels.eq(1).float()).sum(0).squeeze().data.cpu()
        fp = (activations * labels.eq(0).float()).sum(0).squeeze().data.cpu()
        return (tp, fp, num_pos)



    def get_triplets_visualphrase(self):
        """
        Get triplet indices for all visual phrase vocab
        """
        vocab = self.vocab['sro']
        triplets = torch.zeros(len(vocab), 3)
        for j in range(len(vocab)):
            subjname, relname, objname = vocab.idx2word[j].split('-')
            triplets[j, 0] = self.vocab['all'].wordpos2idx[subjname + '_noun']
            triplets[j, 1] = self.vocab['all'].wordpos2idx[objname + '_noun']
            triplets[j, 2] = self.vocab['all'].wordpos2idx[relname + '_verb']

        triplets = triplets.long()
        return triplets


    def get_module_language(self, module_name, embed_size=None, vocab_size=None, num_words=1, normalize=False, finetune=False):
        
        if embed_size==None:
            embed_size= self.embed_size

        if module_name=='word2vec_compositional_2fc':

            input_dim = self.word_embeddings.shape[1]
            module = nn.Sequential( LanguageEmb(self.word_embeddings, finetune=False), \
                                    LanguageProj(embed_size, \
                                                 num_words=num_words, \
                                                 input_dim=input_dim, \
                                                 num_layers=2, \
                                                 hidden_dimension=None,\
                                                 normalize=normalize))

        else:
            print('Language module is unknown')

        return module



    def get_module_vis(self, module_name, embed_size = None, normalize = False, finetune = False, feat_extractor = None):

        assert self.use_precompappearance, 'Activate options use_precompappearance in data loader'
        
        if embed_size == None:
            embed_size = self.embed_size

        if module_name == 'appearanceprecompsubject':

            module = bb.AppearancePrecomp(  embed_size, \
                                            'subject', \
                                            d_appearance=self.d_appearance, \
                                            d_hidden=self.d_hidden, \
                                            normalize=normalize, \
                                            num_layers=self.num_layers)

        elif module_name == 'appearanceprecompobject':

            module = bb.AppearancePrecomp(  embed_size, \
                                            'object', \
                                            d_appearance=self.d_appearance, \
                                            d_hidden=self.d_hidden, \
                                            normalize=normalize, \
                                            num_layers=self.num_layers)

        elif module_name == 'spatialappearanceprecomp':

            module = bb.SpatialAppearancePrecomp(embed_size, \
                                                 d_appearance=self.d_appearance, \
                                                 d_hidden=self.d_hidden, \
                                                 normalize=normalize, \
                                                 num_layers=self.num_layers)
        else:
            print 'Module %s does not exist. Check existing modules or implement it.' % module_name

        return module



    def form_cand_queries_batch(self, batch_input, gram, additional_neg_batch=0):
        """ Sampling queries in batch """
        N = batch_input['pair_objects'].size(0)

        # For each gram s,r,o get unique list of positive labels in the batch
        if gram in ['s','r','o']:

            labels = batch_input['labels_' + gram]

            cat_batch = []
            idx = []
            for j in range(N):

                cats = (labels[j,:]==1).nonzero().data[:,0].tolist()

                count = 0
                cat = cats[0]
                while count < len(cats) and cats[count]>-1:
                    cat = cats[count]
                    if cat not in cat_batch:
                        idx.append(tuple([j,len(cat_batch)]))
                        cat_batch.append(cat)
                    else:
                        idx.append(tuple([j,cat_batch.index(cat)]))
                    count += 1

            # Add negatives at random (later can refine and add hard negatives)
            if additional_neg_batch>0:
                neg_cat_sampled = np.random.randint(0, len(self.vocab[gram]), size=additional_neg_batch) # can be duplicate, it is ok

                # Append the ones that are not positive for any example in the batch
                for neg_cat in neg_cat_sampled:
                    if neg_cat not in cat_batch:
                        cat_batch.append(neg_cat) 
                

            labels_query = np.zeros((N,len(cat_batch)))
            for j in range(len(idx)):
                labels_query[idx[j][0], idx[j][1]] = 1

            cat_batch = Variable(torch.from_numpy(np.array(cat_batch).astype(int)))
            if self.use_gpu:
                cat_batch = cat_batch.cuda()

            query = self.idx_to_vocab[gram].index_select(0, cat_batch)

            labels_query = Variable(torch.from_numpy(labels_query)).type(query.data.type())
            query = query.unsqueeze(0) # (M,1) -> (1,M,1)


            return query, labels_query


        # The triplets for sro are all the positives
        if gram=='sro':

            triplet_cat_batch = np.empty((0,3), dtype=int)
            idx_triplet = []
            for j in range(N):

                sub_cats = (batch_input['labels_s'][j,:]==1).nonzero().data[:,0].tolist()
                obj_cats = (batch_input['labels_o'][j,:]==1).nonzero().data[:,0].tolist()
                rel_cats = (batch_input['labels_r'][j,:]==1).nonzero().data[:,0].tolist()


                # Do not add the triplets containing __background__ -> not in vocab
                for sub_cat in sub_cats:
                    for obj_cat in obj_cats:

                        if sub_cat==0 or obj_cat==0:
                            continue

                        count = 0
                        while count < len(rel_cats) and rel_cats[count]>-1:
                            rel_cat = rel_cats[count]
                            triplet_cat = np.array([sub_cat, rel_cat, obj_cat])
                            idx_triplet_cat_batch = np.where(np.logical_and(triplet_cat_batch[:,0]==triplet_cat[0], \
                                                                            np.logical_and(
                                                                            triplet_cat_batch[:,1]==triplet_cat[1], \
                                                                            triplet_cat_batch[:,2]==triplet_cat[2])))[0]
                            if len(idx_triplet_cat_batch)==0:
                                idx_triplet.append(tuple([j,triplet_cat_batch.shape[0]]))
                                triplet_cat_batch = np.vstack((triplet_cat_batch, triplet_cat))
                            else:
                                idx_triplet.append(tuple([j,idx_triplet_cat_batch[0]]))

                            count += 1


            # Add negatives at random
            if additional_neg_batch>0:

                neg_cat_sampled_sub = np.random.randint(0, len(self.vocab['s']), size=additional_neg_batch)
                neg_cat_sampled_obj = np.random.randint(0, len(self.vocab['o']), size=additional_neg_batch)
                neg_cat_sampled_rel = np.random.randint(0, len(self.vocab['r']), size=additional_neg_batch)
                neg_cat_sampled = np.vstack((neg_cat_sampled_sub, neg_cat_sampled_rel, neg_cat_sampled_obj)).T


                # Append the ones that are not positive for any example in the batch
                for j in range(len(neg_cat_sampled)):

                    idx_batch = np.where(np.logical_and(triplet_cat_batch[:,0]==neg_cat_sampled[j,0], \
                                                        np.logical_and(
                                                        triplet_cat_batch[:,1]==neg_cat_sampled[j,1], \
                                                        triplet_cat_batch[:,2]==neg_cat_sampled[j,2])))[0]

                    if len(idx_batch)==0:
                        triplet_cat_batch = np.vstack((triplet_cat_batch, neg_cat_sampled[j,:]))


            labels_query_sro = np.zeros((N,triplet_cat_batch.shape[0]))
            for j in range(len(idx_triplet)):
                labels_query_sro[idx_triplet[j][0], idx_triplet[j][1]] = 1

            triplet_cat_batch = Variable(torch.from_numpy(triplet_cat_batch))
            if self.use_gpu:
                triplet_cat_batch = triplet_cat_batch.cuda()
            query_sro = torch.cat([ self.idx_to_vocab['s'].index_select(0,triplet_cat_batch[:,0]),\
                                    self.idx_to_vocab['r'].index_select(0,triplet_cat_batch[:,1]),\
                                    self.idx_to_vocab['o'].index_select(0,triplet_cat_batch[:,2])], 1)

            labels_query_sro = Variable(torch.from_numpy(labels_query_sro)).type(query_sro.data.type()) 
            query_sro = query_sro.unsqueeze(0) # (M,3) -> (1,M,3)


            return query_sro, labels_query_sro



    def queries_unigrams(self, words, gram):
        """ Get queries corresponding to objects or predicates """
        if isinstance(words, str) or isinstance(words, unicode):
            words = [words]

        O = len(words)
        queries = torch.zeros(O,1)
        for count, word in enumerate(words):
            cat = self.idx_to_vocab[gram][self.vocab[gram].word2idx[word]].data[0]
            queries[count,:] = cat

        queries = Variable(queries).long()

        if self.use_gpu:
            queries = queries.cuda()

        return queries



    def form_cand_queries_amongvocab(self, batch_input, gram):
        """
        The candidate triplets are all the triplets in vocabulary (of the specific gram)
        NB: this won't be practical for large open vocabulary dataset where you might wish to take a subset of this 
        Output: labels NxM, queries NxMxnum_words
        N: number of candidate pairs
        M: length of vocab[gram]
        num_words: number of words in gram 
        """
        N = batch_input['pair_objects'].size(0)
        tensor_type = batch_input['pair_objects'].long().data.type()

        M = len(self.vocab[gram])
        cats = self.idx_to_vocab[gram].type(tensor_type)
        queries = cats.unsqueeze(0) 
        labels = batch_input['labels_'+gram].type(queries.data.type()) #(N,M)

        return (queries, labels)



    def train_(self, batch_input):
        self.optimizer.zero_grad()
        loss_all = {}
        tp_all = {}
        fp_all = {}
        num_pos_all = {}
        scores, labels = self(batch_input)
        for gram, is_active in self.activated_grams.iteritems():
            if is_active:
                if self.criterions[gram]:
                    loss_all[gram] = self.criterions[gram](scores[gram], labels[gram])
                    activations = (scores[gram] + 1) / 2
                    tp_all[gram], fp_all[gram], num_pos_all[gram] = self.get_statistics(activations, labels[gram])

        loss = 0
        for _, val in loss_all.iteritems():
            loss += val

        loss.backward()
        self.optimizer.step()

        return (loss_all, tp_all,fp_all,num_pos_all)


    def val_(self, batch_input):
        loss_all = {}
        tp_all = {}
        fp_all = {}
        num_pos_all = {}
        scores, labels = self(batch_input)
        for gram, is_active in self.activated_grams.iteritems():
            if is_active:
                if self.criterions[gram]:
                    loss_all[gram] = self.criterions[gram](scores[gram], labels[gram])
                    activations = (scores[gram] + 1) / 2
                    tp_all[gram], fp_all[gram], num_pos_all[gram] = self.get_statistics(activations, labels[gram])

        return (loss_all,tp_all,fp_all,num_pos_all)

    def get_scores(self, batch_input):
        """ Scores produced by independent branches: p(s), p(o), p(r), p(sr), p(ro), p(sro) """
        scores_grams = {}
        scores_grams, _ = self(batch_input)
        for _, gram in enumerate(self.activated_grams):
            if self.activated_grams[gram]:
                scores_grams[gram] = (scores_grams[gram] + 1) / 2

        return scores_grams


    def attach_objectscores_detectors(self, batch_input, scores):
        """ Get the object scores returned by the object detector """
        scores['s'] = batch_input['precompobjectscore'][:,0,:]
        scores['o'] = batch_input['precompobjectscore'][:,1,:]
        return scores


    def form_factors(self, scores, keys):
        """ Form factors from marginals """
        for key in keys:
            grams = key.split('-')
            if np.all([ len(scores[grams[j]]) > 0 for j in range(len(grams)) ]):
                scores[key] = scores[grams[0]].index_select(1, self.idx_sro_to[grams[0]])
                for j in range(1, len(grams)):
                    scores[key] = scores[key] * scores[grams[j]].index_select(1, self.idx_sro_to[grams[j]])

        return scores


    def get_visual_features(self, batch_input, gram):
        """        
        Get visual features for all activated grams:
        vis_feats[gram] : N x d where N is the number of pair of boxes
        """
        vis_feats = self.visual_nets[self.gram_id[gram]](batch_input)

        return vis_feats


    def get_language_features(self, queries, gram):
        """
        Get language features for the queries
        queries : N x M x k where M is the number of queries for each input pair of boxes, k is the number of words
        queries stores the word indices in the big vocab
        Output: language_feats : N x M x d 
        If the queries are shared accross N, then input and output dim is just Mxk, Mxd
        """
        queries_dim = queries.dim()

        if queries_dim==3:
            N = queries.size(0)
            M = queries.size(1)
            num_words = self.num_words[gram]
            queries = queries.view(-1, num_words) # resize (N,M,k) -> (N*M,k)

        language_feats = self.language_nets[self.gram_id[gram]](queries)

        if queries_dim==3:
            language_feats = language_feats.view(N, M, -1)

        return language_feats


    def get_queries(self, batch_input, query_type, gram):

        if query_type=='among_vocab':
            queries, labels = self.form_cand_queries_amongvocab(batch_input, gram)
        elif query_type=='among_batch':
            num_negatives = self.additional_neg_batch
            queries, labels = self.form_cand_queries_batch(batch_input, gram, additional_neg_batch=num_negatives)

        return (queries, labels)

   
    def compute_similarity(self, vis_feats, language_feats, gram):
        """
        Receives vis_feats[gram] of size (N,d), language_feats[gram] of size (N,M,d)
        When there are too many M queries, split the computation to avoid out-of-memory
        """
        queries_dim = language_feats.dim()
        M = language_feats.size(1) if queries_dim==3 else language_feats.size(0)
        N = vis_feats.size(0)
        d = self.embed_size

        # If too many queries, split computation to avoid out-of-memory
        max_num_queries = 10000
        if M <= max_num_queries:
            vis_feats   = vis_feats.unsqueeze(1).expand(N, M, d)
            scores_gram = torch.mul(vis_feats, language_feats)
            scores_gram = scores_gram.sum(2)
            scores      = scores_gram.view(-1, M)
            #scores = torch.matmul(vis_feats, lang_feats.squeeze().transpose(0,1)) #other version

        else:
            scores_gram = [] 
            vis_feats = vis_feats.unsqueeze(1).expand(N, M, d)
            for j in range(M//max_num_queries+1): 
                start_query = j*max_num_queries 
                end_query   = start_query + max_num_queries if start_query + max_num_queries <= M else M
                scores_gram_split = torch.mul(vis_feats[:,start_query:end_query,:], language_feats[:,start_query:end_query,:])
                scores_gram_split = scores_gram_split.sum(2)
                scores_gram.append(scores_gram_split)
            scores = torch.cat([scores_gram_split for scores_gram_split in scores_gram],1)

        return scores


    def forward(self, batch_input):

        scores = {}
        labels = {}

        for _, gram in enumerate(self.activated_grams):
            if self.activated_grams[gram]:

                vis_feats             = self.get_visual_features(batch_input, gram)
                queries, labels[gram] = self.get_queries(batch_input, self.sample_negatives, gram)
                language_feats        = self.get_language_features(queries, gram)
                scores[gram]          = self.compute_similarity(vis_feats, language_feats, gram)

        return (scores, labels)


    def load_pretrained_weights(self, checkpoint):
        """ Load pretrained network """

        model = self.state_dict()
        for key,_ in model.iteritems():
            if key in checkpoint.keys():
                param = checkpoint[key]
                if isinstance(param, Parameter):
                    param = param.data
                model[key].copy_(param)


    """
    Methods for analogy
    """

    def get_gamma(self, input_dim=None):
        if self.gamma=='regadditive':
            reg_network = RegAdditive(self.embed_size, input_dim=3*input_dim, use_proj=True)
        elif self.gamma=='regdeep':
            reg_network = RegDeep(self.embed_size, input_dim=3*input_dim)
        elif self.gamma=='regdeepcond':
            reg_network = RegDeepCond(self.embed_size, input_dim=3*input_dim)
        else:
            reg_network=None
            print('Reg network not specified. Check name.')

        return reg_network



    """
    Form target predictor from source predictor
    """

    def get_lang_precomp_feats(self, gram):

        """ Precompute language features in joint space for gram """
        # Get all predicate queries
        queries = self.idx_to_vocab[gram].data # (P,) : all the predicates 1...P

        queries = Variable(queries)
        if torch.cuda.is_available():
            queries = queries.cuda()

        # Pre-compute language features in joint 'r' space
        print('Precomputing query features in joint space...')
        lang_feats_precomp = self.language_nets[self.gram_id[gram]](queries)

        return lang_feats_precomp



    def get_language_features_analogy(self, queries):
        """
        Input: list of queries Mx3 in vocab[all] indices
        """ 

        language_features_analogy = []


        queries_dim = queries.dim()
        if queries_dim==3:
            M = queries.size(1)
            queries = queries.view(-1,3) # resize (N,M,k) -> (N*M,k)
        elif queries_dim==2:
            M = queries.size(0)


        # For speed-up : pre-compute embedding for target language features
        language_features_target = {}
        language_features_target['s']   = self.get_language_features(queries[:,0].unsqueeze(1), 's').detach()
        language_features_target['r']   = self.get_language_features(queries[:,1].unsqueeze(1), 'r').detach()
        language_features_target['o']   = self.get_language_features(queries[:,2].unsqueeze(1), 'o').detach()
        language_features_target['sro'] = self.get_language_features(queries[:,:], 'sro').detach()


        # Speed-up: pre-compute similarities unigram space between target and all source features
        similarities_precomp = []
        if self.precomp_vp_source_embedding:
            similarities_precomp = self.alpha_s*(torch.matmul(language_features_target['s'], \
                                                                self.lang_feats_precomp_source['s'].transpose(0,1))) + \
                                    (1-self.alpha_r-self.alpha_s)*(torch.matmul(language_features_target['o'], \
                                                                self.lang_feats_precomp_source['o'].transpose(0,1))) + \
                                    self.alpha_r*(torch.matmul(language_features_target['r'], \
                                                                self.lang_feats_precomp_source['r'].transpose(0,1)))
 
        for j in range(M):

            query = queries[j,:]
            language_feats_query = {}
            for gram in language_features_target.keys():
                language_feats_query[gram] = language_features_target[gram][j,:]



            triplet_cat_source, idx_source = self.get_candidates_source(query,\
                                                                    restrict_source_subject   = self.restrict_source_subject,\
                                                                    restrict_source_predicate = self.restrict_source_predicate,\
                                                                    restrict_source_object    = self.restrict_source_object,\
                                                                    num_source_words_common   = self.num_source_words_common)


            # If not source sampled, just use the query embedding
            if triplet_cat_source.shape[0]==0:

                predictor = language_feats_query['sro']

            else:

                # Further threshold source triplets by similarities (could eventually be merged with get_candidates_source
                if len(similarities_precomp)>0:
                    similarities = similarities_precomp[:,idx_source][j]
                    similarities = similarities.data.cpu().numpy()
                else:
                    similarities = self.get_similarities_source(query, triplet_cat_source, \
                                                                sim_method = self.sim_method,\
                                                                alpha_r    = self.alpha_r,\
                                                                alpha_s    = self.alpha_s)

                idx_thresh = self.threshold_similarities_source(similarities, thresh_method = self.thresh_method)


                # If not source sampled, just use query embedding
                if len(idx_thresh)==0:
    
                    predictor = language_feats_query['sro']

                else:
                    triplet_cat_source  = triplet_cat_source[idx_thresh,:]
                    idx_source          = idx_source[idx_thresh]
                    similarities        = similarities[idx_thresh]


                    # If unique source random is activated -> only sample a single source triplet, at random. ATTENTION. Should only be applied at training time. 
                    if self.unique_source_random and self.training:
                        random_id           = np.random.randint(len(idx_thresh))
                        idx_random          = [random_id]
                        triplet_cat_source  = triplet_cat_source[idx_random,:]
                        idx_source          = idx_source[idx_random]
                        similarities        = similarities[idx_random]


                    # Form predictor for source triplets
                    predictor = self.aggregate_predictors_source(query, \
                                                                 language_feats_query,\
                                                                 triplet_cat_source,\
                                                                 idx_source,\
                                                                 similarities, \
                                                                 apply_deformation = self.apply_deformation, \
                                                                 normalize_source  = self.normalize_source, \
                                                                 use_target        = self.use_target)


            language_features_analogy.append(predictor)

        language_features_analogy = torch.stack(language_features_analogy,0)

        if queries_dim==3:
            language_features_analogy = language_features_analogy.view(1, M, -1)


        return language_features_analogy



    def get_candidates_source(self, query, restrict_source_subject=False, restrict_source_predicate=False, restrict_source_object=False, num_source_words_common=2):
        """
        Input : query (sub_cat, rel_cat, obj_cat) in vocab[all] indices
        Output : triplet_cat_source, idx_source 
        """

        # Use precomputed idx source triplets in format[sub_cat_source, rel_cat_source, obj_cat_source] (speed-up) 
        # Remove no_interaction class + sample from triplets occ > minimal mass

        sub_cat_target, rel_cat_target, obj_cat_target = query.data

        triplet_cat_source  = self.triplet_cat_source_common
        idx_source      = np.arange(triplet_cat_source.shape[0], dtype=np.int)


        # Filter depending on subject/object/predicate restriction
        if restrict_source_subject:

            if len(idx_source)>0:
                idx_filter          = np.where(triplet_cat_source[:,0]==sub_cat_target)[0]
                triplet_cat_source  = triplet_cat_source[idx_filter,:]
                idx_source          = idx_source[idx_filter]

        if restrict_source_predicate:

            if len(idx_source)>0:
                idx_filter          = np.where(triplet_cat_source[:,1]==rel_cat_target)[0]
                triplet_cat_source  = triplet_cat_source[idx_filter,:]
                idx_source          = idx_source[idx_filter]

        if restrict_source_object:

            if len(idx_source)>0:
                idx_filter          = np.where(triplet_cat_source[:,2]==obj_cat_target)[0]
                triplet_cat_source  = triplet_cat_source[idx_filter,:]
                idx_source          = idx_source[idx_filter]


        if num_source_words_common>0:
       
            if len(idx_source)>0: 
                num_words_matching = (triplet_cat_source[:,0] == sub_cat_target).astype(int) + \
                                     (triplet_cat_source[:,1] == rel_cat_target).astype(int) + \
                                     (triplet_cat_source[:,2] == obj_cat_target).astype(int)

                idx_filter          = np.where(np.logical_and(num_words_matching>=num_source_words_common, num_words_matching<3))[0]
                triplet_cat_source  = triplet_cat_source[idx_filter,:]
                idx_source          = idx_source[idx_filter]


        return triplet_cat_source, idx_source


    def get_similarities_source(self, query, triplet_cat_source, sim_method='emb_jointspace', alpha_r=0.5, alpha_s=0):

        """ Get back idx cat in vocab_grams to use pre-computed features/sim """
        offset_rel = len(self.vocab['o'])

        sub_cat_target, rel_cat_target, obj_cat_target = query.data
        rel_cat_target -= offset_rel

        sub_cat_source = triplet_cat_source[:,0]
        rel_cat_source = triplet_cat_source[:,1] - offset_rel
        obj_cat_source = triplet_cat_source[:,2]


        """ Compute similarity """

        num_source   = triplet_cat_source.shape[0]
        similarities = np.zeros(num_source, dtype=np.float32)

        if self.sim_precomp:

            for j in range(num_source):

                sim = {}
                sim['s'] = self.sim_precomp['s'][sub_cat_target, sub_cat_source[j]]
                sim['r'] = self.sim_precomp['r'][rel_cat_target, rel_cat_source[j]]
                sim['o'] = self.sim_precomp['o'][obj_cat_target, obj_cat_source[j]] 

                
                similarities[j] = alpha_s*sim['s'] + (1-alpha_r-alpha_s)*sim['o'] + alpha_r*sim['r']


        elif sim_method=='emb_jointspace':

            # Similarity between words of triplet in joint space
            target_sub_emb = self.language_features_precomp['s'][sub_cat_target,:] # (1024,)
            target_rel_emb = self.language_features_precomp['r'][rel_cat_target,:] # (1024,)
            target_obj_emb = self.language_features_precomp['o'][obj_cat_target,:] # (1024,)

            source_subs_emb = self.language_features_precomp['s'][sub_cat_source,:] # (num_source,1024)
            source_rels_emb = self.language_features_precomp['r'][rel_cat_source,:] # (num_source,1024)
            source_objs_emb = self.language_features_precomp['o'][obj_cat_source,:] # (num_source,1024)

            sim_subj = np.matmul(source_subs_emb, target_sub_emb)
            sim_obj  = np.matmul(source_objs_emb, target_obj_emb)
            sim_pred = np.matmul(source_rels_emb, target_rel_emb)

            similarities = alpha_s*sim_subj + (1-alpha_r-alpha_s)*sim_obj + alpha_r*sim_pred

        else:
            print('Similarity method {} is not valid'.format(sim_method))


        return similarities

    def threshold_similarities_source(self, similarities, thresh_method=None):
        """
        Only keep the source triplets whose similarities with target triplet satisfy a threshold
        Output : idx_thresh
        """
        num_source = len(similarities)

        if not thresh_method:
            return np.arange(num_source)

        elif 'top' in thresh_method:
            # Only keep top k source
            k       = int(thresh_method.split('_')[1])
            k       = min(k, num_source) 
            idx_top = np.argsort(similarities)[::-1]
            return idx_top[:k]

        elif 'value' in thresh_method:
            theta    = float(thresh_method.split('_')[1])
            idx_keep = np.where(np.array(similarities)>= theta)[0]
            return idx_keep

        else:
            print('Thresh method {} is not valid'.format(thresh_method))

        return

    def aggregate_predictors_source(self, query, language_features_target, triplet_cat_source, idx_source, similarities, apply_deformation=False, normalize_source=False, use_target=False):

        quadruplets = {}

        """ Get the language embeddings of source triplets """
        if self.precomp_vp_source_embedding:

            for gram in ['s','r','o','sro']:
                quadruplets['source_' + gram] = self.lang_feats_precomp_source[gram][idx_source,:] # already detached

        else:
            queries_sro = Variable(torch.from_numpy(triplet_cat_source).long())

            if torch.cuda.is_available():
                queries_sro = queries_sro.cuda()

            quadruplets['source_sro'] = self.get_language_features(queries_sro, 'sro').detach()
            quadruplets['source_s']   = self.get_language_features(queries_sro[:,0].unsqueeze(1), 's').detach()
            quadruplets['source_r']   = self.get_language_features(queries_sro[:,1].unsqueeze(1), 'r').detach()
            quadruplets['source_o']   = self.get_language_features(queries_sro[:,2].unsqueeze(1), 'o').detach()
            

        """ Get the language embeddings of target triplet """
        quadruplets['target_sro'] = language_features_target['sro'].unsqueeze(0)
        quadruplets['target_s']   = language_features_target['s'].unsqueeze(0)
        quadruplets['target_r']   = language_features_target['r'].unsqueeze(0)
        quadruplets['target_o']   = language_features_target['o'].unsqueeze(0)


        """ Transformed source embedding if needed """
        if apply_deformation:
            deformation   = self.reg_network(quadruplets)
            transformed_source = quadruplets['source_sro'] + deformation
        else:
            transformed_source = quadruplets['source_sro']

        # Normalize before aggregation
        if normalize_source:
            norm               = transformed_source.norm(p=2,dim=1)
            transformed_source = transformed_source.div(norm.unsqueeze(1).expand_as(transformed_source))


        """ Aggregate according to similarity """
        # If use target -> add target to similarity and embedding
        if use_target:
            similarities = np.hstack((similarities, 1.0))
            transformed_source = torch.cat((transformed_source, quadruplets['target_sro']),0)

        # Aggreg with similarity
        similarities = Variable(torch.from_numpy(similarities)).float()

        if torch.cuda.is_available():
            similarities = similarities.cuda()

        similarities_softmax = F.softmax(similarities)
        aggreg_predictor = (similarities_softmax.unsqueeze(1)*transformed_source).sum(0)

        # Normalize again
        if normalize_source:
            norm = aggreg_predictor.norm(p=2)
            aggreg_predictor = aggreg_predictor.div(norm.expand_as(aggreg_predictor))


        return aggreg_predictor


