from __future__ import division
import os
import __init__
import numpy as np
import os.path as osp
from datasets.vrd_api import Vrd
import csv
import pickle
import pdb

# Load vocabulary of triplets
data_path    = '/sequoia/data2/jpeyre/datasets/vrd_iccv17/vrd-dataset'
image_path   = '/sequoia/data2/jpeyre/datasets/vrd_iccv17/vrd-dataset/images'


split = 'trainval'

dset = Vrd(data_path, image_path, split)

# Get set of triplets
triplets_remove = pickle.load(open(osp.join('/sequoia/data2/jpeyre/datasets/unrel_iccv17/unrel-dataset', 'triplet_queries.pkl'), 'rb'))


triplet_cat_remove = np.zeros((len(triplets_remove),3))
for j in range(len(triplets_remove)):
    subjectname, predicate, objectname = triplets_remove[j].split('-')
    triplet_cat_remove[j,0] = dset.vocab_grams['s'].word2idx[subjectname]
    triplet_cat_remove[j,1] = dset.vocab_grams['r'].word2idx[predicate]
    triplet_cat_remove[j,2] = dset.vocab_grams['o'].word2idx[objectname]


# Build a new set of candidates excluding the triplet categories
cand_positives = pickle.load(open(osp.join(data_path, 'cand_positives_' + split + '.pkl'),'rb'))

idx_keep = []
for j in range(cand_positives.shape[0]):
    if j%100000==0:
        print('Done {}/{}'.format(j, cand_positives.shape[0]))
    #obj_cat = cand_positives[j,2]
    im_id   = cand_positives[j,0]
    cand_id = cand_positives[j,1]
    pair_ids            = dset.get_pair_ids(im_id, idx=cand_id)
    gt_sub_cat          = dset.get_gt_classes(im_id, idx=pair_ids[:,0])[0]
    gt_obj_cat          = dset.get_gt_classes(im_id, idx=pair_ids[:,1])[0]
    labels_predicates   = dset.get_labels_predicates(im_id, idx=cand_id)
    gt_rel_cats         = np.where(labels_predicates==1)[1]

    intersect = 0

    for gt_rel_cat in gt_rel_cats:
        idx = np.where( np.logical_and(triplet_cat_remove[:,0] == gt_sub_cat,\
                                        np.logical_and(triplet_cat_remove[:,1] == gt_rel_cat,\
                                                        triplet_cat_remove[:,2] == gt_obj_cat)))[0]        
        if len(idx)>0:
            intersect = 1

    if intersect==0:
        idx_keep.append(j)


cand_positives_keep = cand_positives[idx_keep,:]

pickle.dump(cand_positives_keep, open(osp.join(data_path, 'cand_positives_' + split +'_unrel' +'.pkl'),'wb'))

print('Removed %d/%d positive candidates' %(cand_positives.shape[0]-len(idx_keep), cand_positives.shape[0]))


