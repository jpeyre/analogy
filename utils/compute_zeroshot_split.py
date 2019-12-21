"""
Script to generate the candidates of HICO-DET excluding the 25 zeroshot triplets
NB: you do not need to run this script as we release the candidates
"""
from __future__ import division
import os
import __init__
import numpy as np
import os.path as osp
from datasets.hico_api import Hico
import csv
import pickle

# Load vocabulary of triplets
root_path = './data'
data_path  = '{}/{}'.format(root_path, 'hico')
image_path = '{}/{}/{}'.format(root_path, 'hico', 'images')
cand_dir   = '{}/{}/{}'.format(root_path, 'hico', 'detections')

split = 'trainval' #'train','trainval'
dset = Hico(data_path, image_path, split, cand_dir)

# Get set of triplets
triplets_remove = dset.get_zeroshottriplets()

triplet_cat_remove = []
for l in range(len(triplets_remove)):
    triplet_cat_remove.append(dset.visualphrases.word2idx[triplets_remove[l]])

# Build a new set of candidates excluding the triplet categories
cand_positives = pickle.load(open(osp.join(data_path, 'cand_positives_' + split + '.pkl'),'rb'))

idx_keep = []
for j in range(cand_positives.shape[0]):
    if j%100000==0:
        print('Done {}/{}'.format(j, cand_positives.shape[0]))
    im_id = cand_positives[j,0]
    cand_id = cand_positives[j,1]
    # Load the gt label of visualphrase
    triplet_cat = np.where(dset.get_labels_visualphrases(im_id, cand_id))[1]
    intersect = np.intersect1d(triplet_cat, triplet_cat_remove)
    if len(intersect)==0:
        idx_keep.append(j)

cand_positives_keep = cand_positives[idx_keep,:]

pickle.dump(cand_positives_keep, open(osp.join(data_path, 'cand_positives_' + split +'_zeroshottriplet.pkl'),'wb'))

print('Removed %d positive candidates' %(cand_positives.shape[0]-len(idx_keep)))
