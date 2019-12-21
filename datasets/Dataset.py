import numpy as np
from datasets.utils import get_union_boxes, jitter_boxes, flip_horizontal, nms, Vocabulary, box_integer 
import os.path as osp
import numbers
import cv2
import scipy.sparse as sparse
import pdb
import math

class BaseDataset(object):
    """ This superclass contain all methods shared across different datasets """
    def __init__(self):
        pass


    def get_idx_triplets_grams(self, triplets):
        """
        Similar to get_idx_between_vocab except that we provide as input the set of triplets we are interested in
        If the triplet is out of vocab -> would put idx to -1
        """
        keys = {'s':[], 'o':[], 'r':[], 'sr':[], 'ro':[], 'sro':[]}
        idx_triplets_to = {}

        for key in keys:
            vocab_gram = self.vocab_grams[key]
            idx = -np.ones((len(triplets),))

            if key=='s':
                for t, triplet in triplets.idx2word.iteritems():
                    triplet_s = triplet.split('-')[0]
                    if triplet_s in vocab_gram.words():
                        idx[t] = vocab_gram.word2idx[triplet_s]

            if key=='o':
                for t, triplet in triplets.idx2word.iteritems():
                    triplet_o = triplet.split('-')[2]
                    if triplet_o in vocab_gram.words():
                        idx[t] = vocab_gram.word2idx[triplet_o]

            if key=='r':
                for t, triplet in triplets.idx2word.iteritems():
                    triplet_r = triplet.split('-')[1]
                    if triplet_r in vocab_gram.words():
                        idx[t] = vocab_gram.word2idx[triplet_r]

            if key=='sr':
                for t, triplet in triplets.idx2word.iteritems():
                    triplet_sr = triplet.split('-')
                    triplet_sr = '-'.join([triplet_sr[0],triplet_sr[1]])
                    if triplet_sr in vocab_gram.words():
                        idx[t] = vocab_gram.word2idx[triplet_sr]

            if key=='ro':
                for t, triplet in triplets.idx2word.iteritems():
                    triplet_ro = triplet.split('-')
                    triplet_ro = '-'.join([triplet_ro[1],triplet_ro[2]])
                    if triplet_ro in vocab_gram.words():
                        idx[t] = vocab_gram.word2idx[triplet_ro]

            if key=='sro':
                for t, triplet_sro in triplets.idx2word.iteritems():
                    if triplet_sro in vocab_gram.words():
                        idx[t] = vocab_gram.word2idx[triplet_sro]

            idx_triplets_to[key] = idx


        return idx_triplets_to


    def get_idx_in_vocab(self, vocab_grams, vocab_all, use_sro=True):
        """
        Get correspondencies of vocab_grams in vocab['all'] containing all nouns/verbs
        For this check both word + part-of-speech
        """
        #keys = {'s','o','r','sr','ro','sro'} # for bigram experiment uncomment (todo: add option)
        keys = {'s','o','r','sro'} # for experiment coco-a uncommnet
        num_words = {'s':1,'o':1,'r':1,'sr':2,'ro':2,'sro':3}
        idx_in_vocab = {}

        for key in keys:

            if key=='sro' and not use_sro:
                continue

            vocab_gram = vocab_grams[key]
            num_words_gram = num_words[key]     

            idx = -np.ones((len(vocab_gram),num_words_gram))
            for i in vocab_gram.idx2word.keys():
                # Split the words
                wordpos_gram = vocab_gram.idx2wordpos[i].split('_')
                word_gram = wordpos_gram[0].split('-')
                pos_gram = wordpos_gram[1].split('-')                


                for l in range(num_words_gram):
                    wordpos = '_'.join([word_gram[l], pos_gram[l]])

                    if wordpos in vocab_all.wordpos2idx:
                        idx[i,l] = vocab_all.wordpos2idx[wordpos]

            idx_in_vocab[key] = idx

        return idx_in_vocab


    def get_idx_between_vocab(self, vocab_triplet, vocab_grams, use_sro=True):
        """
        Get idx for correspondencies between vocab of each gram and vocab_sro
        """
        keys = []
        idx_sro_to = {}
        for key in vocab_grams.keys():
            if len(vocab_grams[key])>0:
                keys.append(key)
                idx_sro_to[key] = -np.ones((len(vocab_triplet),))

        #keys = {'s', 'o', 'r', 'sr', 'ro', 'sro'}
        for t, triplet in vocab_triplet.idx2word.iteritems():

            subjectname, predicate, objectname = triplet.split('-')
            
            idx_sro_to['s'][t] = vocab_grams['s'].word2idx[subjectname]
            idx_sro_to['o'][t] = vocab_grams['o'].word2idx[objectname]
            idx_sro_to['r'][t] = vocab_grams['r'].word2idx[predicate]

            if 'sro' in keys and use_sro:
                idx_sro_to['sro'][t] = vocab_grams['sro'].word2idx[triplet]


            if 'sr' in keys:
                bigram_sr = '-'.join([subjectname, predicate])
                idx_sro_to['sr'][t] = vocab_grams['sr'].word2idx[bigram_sr]

            if 'ro' in keys:
                bigram_ro = '-'.join([predicate, objectname])
                idx_sro_to['ro'][t] = vocab_grams['ro'].word2idx[bigram_ro]


        return idx_sro_to


    def get_test_candidates(self, use_gt=False, thresh_file=None, nms_thresh=0.5, sort_by_im=False):
        """
        Form all candidates for test
        Output : candidates: Nx2 [im_id, cand_id] 
        """

        # By default concatenate the candidates from different images in same array (to speed-up test loader)
        candidates = {}
        for im_id in self.image_ids:
            if use_gt:
                idx_valid = self.filter_pairs_gt(im_id)
            else:
                idx_valid = self.filter_pairs_cand(im_id)

                # Filter detections by their scores
                if thresh_file:
                    idx_valid = self.filter_by_scores(im_id, self.dets_thresh, idx=idx_valid)

                # Filter detections by NMS
                if nms_thresh!=0.5:
                    idx_valid = self.filter_by_nms(im_id, nms_thresh, idx=idx_valid)


            if len(idx_valid)>0:
                candidates[int(im_id)] = idx_valid.astype(np.int32)

        """ Stack all candidates from all images together """
        all_candidates = np.zeros((1000000,2), dtype=np.int32)
        count = 0
        for im_id in candidates.keys():

            if count+candidates[im_id].shape[0] >= all_candidates.shape[0]:
                new_array = np.zeros((1000000+all_candidates.shape[0],2))
                new_array[:count,:] = np.array(all_candidates[:count,:])
                all_candidates = new_array

            all_candidates[count:count+candidates[im_id].shape[0],0] = im_id
            all_candidates[count:count+candidates[im_id].shape[0],1] = candidates[im_id]
            count += candidates[im_id].shape[0]
            #all_candidates = np.vstack((all_candidates, np.hstack((im_id*np.ones((len(candidates[im_id]),1)), candidates[im_id][:,None])) ))
        all_candidates = all_candidates.astype(np.int32)
        all_candidates = all_candidates[:count,:]

        return all_candidates
 

    def get_training_candidates(self, use_gt=False, add_gt=True, thresh_file=None):
        """
        Form all relevant candidates for training : either sample in GT, either sample in candidate detections
        Don't sample negatives from "ignore" humans
        Output: cand_positives : Nx2 [im_id, cand_id] where cand_id is the candidate pair indexed in image
                cand_negatives: Nx2 [im_id, cand_id]
        use_gt = set to True if use groundtruth pairs, by default using candidates
        add_gt = put to False if you false to use the groundtruth pairs additionally to candidates. Default is True. 
        """

        if use_gt==True:
            assert add_gt==False, 'Set add_gt to False if use_gt is True'

        large_number = 1000000

        cand_positives = np.zeros((large_number,4), dtype=np.int32)
        cand_negatives = np.zeros((large_number,4), dtype=np.int32)

        print('Get training candidates')
        count_pos = 0
        count_neg = 0

        for count_im, im_id in enumerate(self.image_ids):

            if count_im%1000==0:
                print('Done {}/{}'.format(count_im, len(self.image_ids)))

            if use_gt:
                idx_valid = self.filter_pairs_gt(im_id)
            else:
                if not add_gt:
                    idx_valid = self.filter_pairs_cand(im_id)
                else:
                    idx_valid = np.arange(len(self.db[im_id]['is_gt_pair'])) # keeping both GT and candidates


            # Filter detections by their scores
            if thresh_file:
                idx_valid = self.filter_by_scores(im_id, self.dets_thresh, idx=idx_valid)

            # Positives candidates
            idx_pos = self.filter_pairs_pos(im_id, idx=idx_valid)

            # Get negatives
            idx_neg = self.filter_pairs_neg(im_id, idx=idx_valid)

            # Append subject category 
            sub_id_pos = self.get_pair_ids(im_id, idx=idx_pos)[:,0]
            sub_cat_pos = self.get_classes(im_id, idx=sub_id_pos)
            sub_id_neg = self.get_pair_ids(im_id, idx=idx_neg)[:,0]
            sub_cat_neg = self.get_classes(im_id, idx=sub_id_neg)

            # Append object category to filter negatives
            obj_id_pos = self.get_pair_ids(im_id, idx=idx_pos)[:,1]
            obj_cat_pos = self.get_classes(im_id, idx=obj_id_pos)
            obj_id_neg = self.get_pair_ids(im_id, idx=idx_neg)[:,1]
            obj_cat_neg = self.get_classes(im_id, idx=obj_id_neg)

            # Add the positives
            if count_pos + len(idx_pos) >= cand_positives.shape[0]:
                new_array = np.zeros((1000000+cand_positives.shape[0],4))
                new_array[:count_pos,:] = np.array(cand_positives[:count_pos,:])
                cand_positives = new_array

            if count_neg + len(idx_neg) >= cand_negatives.shape[0]:
                new_array = np.zeros((1000000+cand_negatives.shape[0],4))
                new_array[:count_neg,:] = np.array(cand_negatives[:count_neg,:])
                cand_negatives = new_array
        
            cand_positives[count_pos:count_pos+len(idx_pos),0] = im_id
            cand_positives[count_pos:count_pos+len(idx_pos),1] = np.array(idx_pos)
            cand_positives[count_pos:count_pos+len(idx_pos),2] = np.array(sub_cat_pos)
            cand_positives[count_pos:count_pos+len(idx_pos),3] = np.array(obj_cat_pos)

            cand_negatives[count_neg:count_neg+len(idx_neg),0] = im_id
            cand_negatives[count_neg:count_neg+len(idx_neg),1] = np.array(idx_neg)
            cand_negatives[count_neg:count_neg+len(idx_neg),2] = np.array(sub_cat_neg)
            cand_negatives[count_neg:count_neg+len(idx_neg),3] = np.array(obj_cat_neg)

             
            count_pos += len(idx_pos)
            count_neg += len(idx_neg)

        cand_positives = cand_positives[:count_pos,:].astype(np.int32)
        cand_negatives = cand_negatives[:count_neg,:].astype(np.int32)


        return cand_positives, cand_negatives



    def filter_by_nms(self, im_id, nms_thresh, idx=None):

        idx_pair = self.get_pair_ids(im_id, idx=idx)

        # Apply NMS on unique human boxes
        idx_human_unique, indices = np.unique(idx_pair[:,0], return_inverse=True)
        idx_human_unique_valid = np.zeros(len(idx_human_unique), dtype=np.bool)
        idx_human_valid = np.zeros(len(indices), dtype=np.bool)

        subject_scores = self.get_scores(im_id, idx=idx_human_unique)
        subject_boxes = self.get_boxes(im_id, idx=idx_human_unique)
        dets = np.hstack((subject_boxes, subject_scores[:,None]))
        keep = nms(dets, nms_thresh)
        idx_human_unique_valid[keep] = 1
        idx_human_valid = idx_human_unique_valid[indices] # should return boolean of dimension number of humans

        # Filter objects by scores: by category
        idx_object_unique, indices = np.unique(idx_pair[:,1], return_inverse=True)
        idx_object_unique_valid = np.zeros(len(idx_object_unique), dtype=np.bool)
        idx_object_valid = np.zeros(len(indices),dtype=np.bool)

        object_scores = self.get_scores(im_id, idx=idx_object_unique)
        object_boxes = self.get_boxes(im_id, idx=idx_object_unique)
        object_classes = self.get_classes(im_id, idx=idx_object_unique)
        object_classes_unique = np.unique(object_classes)
        idx_object_unique_valid = np.zeros(len(idx_object_unique), dtype=np.bool)

        for c in range(len(object_classes_unique)):
            id_cat = object_classes_unique[c]
            idx_cat = np.where(object_classes==id_cat)[0]
            dets = np.hstack((object_boxes[idx_cat,:], object_scores[idx_cat,None]))
            keep = nms(dets, nms_thresh)
            keep = idx_cat[keep]
            idx_object_unique_valid[keep] = 1
            idx_object_valid = np.logical_or(idx_object_valid, idx_object_unique_valid[indices])

        idx_pair_valid = np.where(np.logical_and(idx_human_valid, idx_object_valid)==1)[0]
        idx_pair_valid = idx[idx_pair_valid]

        return idx_pair_valid




    def filter_objects_human(self, im_id, idx=None):
        """
        Return idx of human objects in image
        """
        classes = self.db[im_id]['obj_classes']
        if idx is None:
            idx_select = np.arange(len(classes))
        else:
            if isinstance(idx, numbers.Number):
                idx_select = np.array([idx])

        idx_human = np.where(classes[idx_select]==1)[0]
        idx_human = idx_select[idx_human]

        return idx_human


    def filter_objects_gt(self, im_id, idx=None):
        """
        Return idx of groundtruth objects in image
        """
        if idx is None:
            idx_gt = np.where(self.db[im_id]['is_gt']==1)[0]
        else:
            if isinstance(idx, numbers.Number):
                idx = np.array([idx])
            idx_gt = np.where(self.db[im_id]['is_gt'][idx]==1)[0]
            idx_gt = idx[idx_gt]
        return idx_gt


    def filter_objects_cand(self, im_id, idx=None):
        """
        Return idx of groundtruth objects in image
        """
        if idx is None:
            idx_gt = np.where(self.db[im_id]['is_gt']==0)[0]
        else:
            if isinstance(idx, numbers.Number):
                idx = np.array([idx])
            idx_gt = np.where(self.db[im_id]['is_gt'][idx]==0)[0]
            idx_gt = idx[idx_gt]
        return idx_gt


    def filter_pairs_gt(self, im_id, idx=None):
        """
        Return idx of groundtruth pairs in image
        """
        if idx is None:
            idx_gt = np.where(self.db[im_id]['is_gt_pair']==1)[0]
        else:
            if isinstance(idx, numbers.Number):
                idx = np.array([idx])
            idx_gt = np.where(self.db[im_id]['is_gt_pair'][idx]==1)[0]
            idx_gt = idx[idx_gt]
        return idx_gt


    def filter_pairs_cand(self, im_id, idx=None):
        """
        Return idx of candidate pairs in image
        """
        if idx is None:
            idx_cand = np.where(self.db[im_id]['is_gt_pair']==0)[0]
        else:
            if isinstance(idx, numbers.Number):
                idx = np.array([idx])
            idx_cand = np.where(self.db[im_id]['is_gt_pair'][idx]==0)[0]
            idx_cand = idx[idx_cand]
        return idx_cand


    def filter_pairs_pos(self, im_id, idx=None):
        """
        Return idx of positive pairs in image
        """
        labels = self.get_labels_predicates(im_id)
        if idx is None:
            idx_pos = np.where(np.any(labels[:,1:], axis=1))[0]
        else:
            if isinstance(idx, numbers.Number):
                idx = np.array([idx])
            idx_pos = np.where(np.any(labels[idx,1:], axis=1))[0]
            idx_pos = idx[idx_pos]

        return idx_pos


    def filter_pairs_neg(self, im_id, idx=None):
        """
        Return idx of negative pairs in image
        """
        labels = self.get_labels_predicates(im_id)
        if idx is None:
            idx_neg = np.where(labels[:,0]==1)[0]
        else:
            if isinstance(idx, numbers.Number):
                idx = np.array([idx])
            idx_neg = np.where(labels[idx,0]==1)[0]
            idx_neg = idx[idx_neg]
        return idx_neg


    def filter_by_scores(self, im_id, dets_thresh, idx=None):

        idx_pair = self.get_pair_ids(im_id, idx=idx)

        # Filter humans by scores
        subject_scores = self.get_scores(im_id, idx=idx_pair[:,0])
        idx_human_valid = subject_scores >= dets_thresh[0]

        # Filter objects by scores
        object_scores = self.get_scores(im_id, idx=idx_pair[:,1])
        object_classes = self.get_classes(im_id, idx=idx_pair[:,1])
        objs_thresh = dets_thresh[object_classes-1]
        idx_object_valid = object_scores >= objs_thresh

        idx_pair_valid = np.where(np.logical_and(idx_human_valid, idx_object_valid)==1)[0]
        idx_pair_valid = idx[idx_pair_valid]

        return idx_pair_valid


    def load_pair_objects(self, im_id, cand_id):
        """
        Load object boxes for all candidate pairs
        Input:  im_id
                cand_id : index of candidate pair in image
        Output:
                objects (1,2,6) : [x1,y1,x2,y2,obj_cat,obj_score]
        """
        pair_ids = self.get_pair_ids(im_id, idx=cand_id)
        pair_objects = np.zeros((pair_ids.shape[0],2,6))
        pair_objects[:,0,:4] = self.get_boxes(im_id, idx=pair_ids[:,0])
        pair_objects[:,0,4]  = self.get_classes(im_id, idx=pair_ids[:,0])
        pair_objects[:,0,5]  = self.get_scores(im_id, idx=pair_ids[:,0])
        pair_objects[:,1,:4] = self.get_boxes(im_id, idx=pair_ids[:,1])
        pair_objects[:,1,4]  = self.get_classes(im_id, idx=pair_ids[:,1])
        pair_objects[:,1,5]  = self.get_scores(im_id, idx=pair_ids[:,1])

        if self.jittering:
            width, height = self.image_size(im_id)
            pair_objects[:,0,:4] = jitter_boxes(pair_objects[:,0,:4], width, height)
            pair_objects[:,1,:4] = jitter_boxes(pair_objects[:,1,:4], width, height)

            # Horizontal flip
            if np.random.binomial(1,0.5):
                pair_objects[:,0,:4] = flip_horizontal(pair_objects[:,0,:4], width, height)
                pair_objects[:,1,:4] = flip_horizontal(pair_objects[:,0,:4], width, height)


        return pair_objects


    def build_vocab(self, nouns, predicates):
        """ Build joint vocabulary of nouns and predicates """
        vocab = Vocabulary()

        # Add nouns
        for word in nouns.words():
            vocab.add_word(word, 'noun')

        # Add predicates
        for word in predicates.words():
            vocab.add_word(word, 'verb')

        return vocab


    def image_size(self, im_id):
        width = self.db[im_id]['width']
        height = self.db[im_id]['height']

        return width, height


    def get_obj_id(self, im_id, idx=None):
        """ Return : (N,) annotation id of objects """
        obj_id = self.db[im_id]['obj_id']
        if idx is not None:
            if isinstance(idx, numbers.Number):
                idx = np.array([idx])
            obj_id = obj_id[idx]
        return obj_id


    def get_cand_id(self, im_id, idx=None):
        """ Return : (N,) annotation id of pairs """
        cand_id = self.db[im_id]['cand_id']
        if idx is not None:
            if isinstance(idx, numbers.Number):
                idx = np.array([idx])
            cand_id = cand_id[idx]
        return cand_id


    def get_boxes(self, im_id, idx=None):
        """ Return : (N,4) : [xmin, ymin, xmax, ymax] """
        bboxes = self.db[im_id]['boxes']
        if idx is not None:
            if isinstance(idx, numbers.Number):
                idx = np.array([idx])
            bboxes = bboxes[idx,:]
        return bboxes

    def get_classes(self, im_id, idx=None):
        """
        Detected classes (by object detector)
        Return : (N,) : [obj_cat]
        """
        classes = self.db[im_id]['obj_classes']
        if idx is not None:
            if isinstance(idx, numbers.Number):
                idx = np.array([idx])
            classes = classes[idx]
        return classes

    def get_gt_classes(self, im_id, idx=None):
        """
        Groundtruth class 
        Return : (N,) : [obj_gt]
        """
        classes = self.db[im_id]['obj_gt_classes']
        if idx is not None:
            if isinstance(idx, numbers.Number):
                idx = np.array([idx])
            classes = classes[idx]
        return classes


    def get_scores(self, im_id, idx=None):
        """ Return : (N,) : [obj_scores] """
        scores = self.db[im_id]['obj_scores']
        if idx is not None:
            if isinstance(idx, numbers.Number):
                idx = np.array([idx])
            scores = scores[idx]
        return scores


    def get_pair_ids(self, im_id, idx=None):
        """ Return : (N,2) : [subject_id, object_id] : idx of subjects, objects in pairs """
        pair_ids = self.db[im_id]['pair_ids']
        if idx is not None:
            if isinstance(idx, numbers.Number):
                idx = np.array([idx])
            pair_ids = pair_ids[idx,:]
        return pair_ids

    def get_boxes_union(self, im_id, idx=None):
        """
        Return : (N,4) : [x1,y1,x2,y2] union boxes for pairs
        """
        pair_ids = self.get_pair_ids(im_id, idx)
        subject_boxes = self.db[im_id]['boxes'][pair_ids[:,0],:]
        object_boxes = self.db[im_id]['boxes'][pair_ids[:,1],:]
        xu = np.minimum(subject_boxes[:,0], object_boxes[:,0])
        yu = np.minimum(subject_boxes[:,1], object_boxes[:,1])
        xu_end = np.maximum(subject_boxes[:,2], object_boxes[:,2])
        yu_end = np.maximum(subject_boxes[:,3], object_boxes[:,3])
        union_boxes = np.stack((xu, yu, xu_end, yu_end), axis=1)
        return union_boxes

    def get_labels_predicates(self, im_id, idx=None):
        """
        Return : (N,num_predicates) : binary labels
        """
        labels = self.db[im_id]['labels_r']
        if idx is not None:
            if isinstance(idx, numbers.Number):
                idx = np.array([idx])
            labels = labels[idx,:]
        labels = labels.toarray()

        return labels


    def get_labels_objects(self, im_id, idx=None):
        """
        Return : (N,num_objects) : binary labels
        """
        pair_ids = self.get_pair_ids(im_id, idx=idx)
        N = pair_ids.shape[0]
        obj_cat = self.get_gt_classes(im_id, idx=pair_ids[:,1])
        labels = np.zeros((N,len(self.classes)))
        labels[np.arange(N), obj_cat] = 1

        return labels

    def get_labels_subjects(self, im_id, idx=None):
        """
        Return : (N,num_objects) : binary labels
        """
        pair_ids = self.get_pair_ids(im_id, idx=idx)
        N = pair_ids.shape[0]
        sub_cat = self.get_gt_classes(im_id, idx=pair_ids[:,0])
        labels = np.zeros((N,len(self.classes)))
        labels[np.arange(N), sub_cat] = 1

        return labels


    def get_labels_subjectpredicates(self, im_id, idx=None):
        """
        Return :(N, num_subjectpredicates): subject is always person
        """
        pair_ids = self.get_pair_ids(im_id, idx=idx)
        sub_ids = pair_ids[:,0]
        labels_subjectpredicates = self.db[im_id]['labels_sr'][sub_ids,:].toarray()

        return labels_subjectpredicates


    def get_labels_objectpredicates(self, im_id, idx=None):
        """
        Return :(N, num_objectpredicates)
        """
        pair_ids = self.get_pair_ids(im_id, idx=idx)
        obj_ids = pair_ids[:,1]
        labels_objectpredicates = self.db[im_id]['labels_ro'][obj_ids].toarray()

        return labels_objectpredicates


