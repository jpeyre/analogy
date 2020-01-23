"""
Evaluate for retrieval of unseen triplets of UnRel
"""

from __future__ import division
import __init__
import tensorboard_logger 
import torch
import torch.nn as nn
from torch.autograd import Variable
import pdb
import numpy as np
import time
from utils import Parser
import argparse
import os.path as osp
import cPickle as pickle
from datasets.BaseLoader import TestSampler
from networks import models
import scipy.io as sio
import yaml
import warnings
warnings.filterwarnings("ignore")
import os
import torch.nn.functional as F
import csv


""" Parsing options """

args = argparse.ArgumentParser()
parser = Parser(args)
opt = parser.make_options()


def print_similarities(target_triplets, queries_sro):

    for count,target_triplet in enumerate(target_triplets):

        query = queries_sro[count,:]

        triplet_cat_source, idx_source = model.get_candidates_source(query,\
                                                                    restrict_source_subject   = opt.restrict_source_subject,\
                                                                    restrict_source_predicate = opt.restrict_source_predicate,\
                                                                    restrict_source_object    = opt.restrict_source_object,\
                                                                    num_source_words_common   = opt.num_source_words_common)

        similarities = model.get_similarities_source(query, triplet_cat_source, \
                                                                sim_method = opt.sim_method,\
                                                                alpha_r    = opt.alpha_r,\
                                                                alpha_s    = opt.alpha_s)


        idx_thresh = model.threshold_similarities_source(similarities, \
                                                                thresh_method = opt.thresh_method)


        triplet_cat_source = triplet_cat_source[idx_thresh,:]
        similarities = similarities[idx_thresh]


        filename_out = osp.join(sim_dir, 'similarities_{}_{}_{}_{}.csv'.format(\
                                        opt.cand_test,\
                                        opt.test_split,\
                                        opt.epoch_model,\
                                        target_triplet))


        # Sort by decreasing similarity
        idx_sort = np.argsort(similarities)[::-1]

        with open(filename_out, 'wb') as g:
            writer = csv.writer(g)
            writer.writerow(['source_triplet', 'similarity','occ_train'])

            for l in range(len(idx_sort)):
                s = idx_sort[l]

                # Indices in all vocab
                sub_cat_source, rel_cat_source, obj_cat_source = triplet_cat_source[s,:]

                if opt.use_analogy:
                    source_triplet = '-'.join([ model.vocab['all'].idx2word[sub_cat_source],\
                                                model.vocab['all'].idx2word[rel_cat_source],\
                                                model.vocab['all'].idx2word[obj_cat_source]])


                else:
                    source_triplet = '-'.join([ model.vocab['s'].idx2word[sub_cat_source],\
                                                model.vocab['r'].idx2word[rel_cat_source],\
                                                model.vocab['o'].idx2word[obj_cat_source]])


                source_sim = '{:.3f}'.format(similarities[s])


                # Occurrence
                occ_train = int(model.triplet_mass[idx_source[s]])

                # Csv
                writer.writerow([source_triplet, source_sim, occ_train])



def get_scores(loader, keys, aggregation=None):

    large_number = 300000 # max number of candidates 

    res_scores = {}
    res_scores['r'] = np.zeros((large_number,70+1)) # add rel_id column
    res_scores['sro'] = np.zeros((large_number,len(target_triplets)+1)) # add rel_id column
    res_scores['s'] = np.zeros((large_number,1+1)) # add rel_id column
    res_scores['o'] = np.zeros((large_number,1+1)) # add rel_id column


    start_time = time.time()
    count = 0

    for batch_idx, batch_input in enumerate(loader):

        # Get boxes and object class
        im_ids = batch_input['cand_info'][:,0].numpy()
        cand_ids = batch_input['cand_info'][:,1].numpy()
        pair_objects = batch_input['pair_objects'].numpy()
        obj_classes = pair_objects[:,1,4]
        sub_classes = pair_objects[:,0,4]

        # Get rel_id
        rel_id_iccv = np.zeros((len(im_ids),)).astype(int)
        for j in range(len(im_ids)):
            im_id = im_ids[j]
            cand_id = cand_ids[j]
            rel_id_iccv[j] = dset.get_rel_id_iccv(im_id, cand_id)[0]

        for key in keys:
            res_scores[key][count:count+len(im_ids),0] = rel_id_iccv

        for inpt in batch_input.keys():
            if opt.use_gpu:
                batch_input[inpt] = Variable(batch_input[inpt].cuda())
            else:
                batch_input[inpt] = Variable(batch_input[inpt])

        # Get the visual features
        vis_feats = {}
        for _,gram in enumerate(model.activated_grams):
            if model.activated_grams[gram]:
                vis_feats[gram] = model.get_visual_features(batch_input, gram) #(N,1024)

        # Forward scores
        scores = model.get_scores(batch_input) # get scores of grams 's','r','o','sr','ro','sro'. Between [0,1]
        if isinstance(scores,tuple):
            scores = scores[0]


        """ Attach object scores : else, object scores by object branches (to try actually) """
        if opt.use_objscoreprecomp:
            scores = model.attach_objectscores_detectors(batch_input, scores)


        # Fill 'r' scores (remove no_interaction)
        res_scores['r'][count:count+len(im_ids),1:] = scores['r'].data.cpu().numpy()[:,1:]

        # Fill 's','o' scores
        res_scores['o'][count:count+len(im_ids),1] = scores['o'].data.cpu().numpy()[np.arange(len(obj_classes)), obj_classes.astype(int)]
        res_scores['s'][count:count+len(im_ids),1] = scores['s'].data.cpu().numpy()[np.arange(len(sub_classes)), sub_classes.astype(int)]

        # Now get scores for sro
        if 'sro' in keys:
            scores_sro = torch.matmul(vis_feats['sro'], lang_feats_precomp_sro.transpose(0,1))
            scores_sro = F.sigmoid(scores_sro) # could eventually throw away unless you need to aggregate with sro
            scores_sro = scores_sro.data.cpu().numpy()

            for triplet_id, target_triplet in enumerate(target_triplets):

                res_scores['sro'][count:count+len(im_ids),triplet_id+1] = np.array(scores_sro[:,triplet_id])


        # Increment count
        count += len(im_ids)

        #del batch_input

        if batch_idx % 100 ==0:
            print('Done [{}/{}] in {:.2f} sec'.format(batch_idx, len(loader), time.time()-start_time))
            start_time = time.time()


    for key in keys:
        res_scores[key] = res_scores[key][:count,:]

    return res_scores


####################
""" Data loaders """
####################

if opt.cand_test=='gt':
    use_gt = True
else:
    use_gt = False



def load_dataset(data_name, cand_test):

    data_path = '/sequoia/data2/jpeyre/datasets/{}_iccv17/{}-dataset'.format(data_name, data_name)
    image_path = '/sequoia/data2/jpeyre/datasets/{}_iccv17/{}-dataset/images'.format(data_name, data_name)


    if data_name=='vrd':
        from datasets.vrd_api import Vrd as Dataset
    elif data_name=='unrel':
        from datasets.unrel_api import Unrel as Dataset


    dset = Dataset( data_path,\
                    image_path, \
                    opt.test_split, \
                    cand_dir = '',\
                    cand_test = cand_test,\
                    thresh_file = None, \
                    use_gt = use_gt, \
                    add_gt = False, \
                    train_mode = False, \
                    jittering = False, \
                    nms_thresh = 1.0,\
                    store_ram = [],\
                    l2norm_input = opt.l2norm_input)


     dset_loader = TestSampler(dset,\
                    use_precompappearance   = opt.use_precompappearance, \
                    use_precompobjectscore  = opt.use_precompobjectscore)
    

    loader = torch.utils.data.DataLoader(dset_loader, batch_size=8, shuffle=False, num_workers=0, collate_fn=dset_loader.collate_fn)

    return dset, loader 


# Init with vrd
dset, loader = load_dataset('vrd', 'gt-candidates')


####################################
""" Get the triplets to retrieve """
####################################

subset_test = 'unrel'

target_triplets = pickle.load(open('/sequoia/data2/jpeyre/datasets/unrel_iccv17/unrel-dataset/triplet_queries.pkl', 'rb')) # Unrel queries in VG vocab


print('Computing retrieval on {} triplet queries'.format(len(target_triplets)))

##################
""" Load model """
##################

logger_path = osp.join(opt.logger_dir, opt.exp_name)
opt = parser.get_opts_from_dset(opt, dset)


# Load model
print('Loading model')
model = models.get_model(opt)
checkpoint = torch.load(osp.join(logger_path, 'model_' + opt.epoch_model + '.pth.tar'), map_location=lambda storage, loc: storage)
model.load_state_dict(checkpoint['model'])
model.eval()


if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)

if torch.cuda.is_available():
    model.cuda()



###########################
""" Precomp for speedup """
###########################

if opt.use_analogy:

    model.precomp_language_features() # pre-compute unigram emb
    model.precomp_sim_tables() # pre-compute similarity tables for speed-up


# Target queries indices
queries_sro, triplet_queries_idx = model.precomp_target_queries(triplet_queries)


# Pre-compute language features in joint 'sro' space
if opt.embedding_type=='target':
    print('Precomputing query features in joint space...')
    lang_feats_precomp_sro = model.get_language_features(queries_sro, 'sro')
elif opt.use_analogy:
    print('Precomputing query features in joint space with analogy...')
    lang_feats_precomp_sro = model.get_language_features_analogy(queries_sro)


################
""" Evaluate """
################

keys = ['r','sro','s','o']

name_dir = 'scores_retrieval_' + subset_test + '_' + opt.embedding_type


save_dir = parser.get_res_dir(opt, name_dir)
sim_dir = parser.get_res_dir(opt, 'similarities_' + subset_test + '_' + opt.embedding_type)

# Print source triplets and simmilarities
if not opt.embedding_type=='target':
    print_similarities(target_triplets, queries_sro)


print('Begin evaluation')
datasets = ['unrel','vrd']
num_cand_theoric = {'vrd_candidates':290974,'vrd_gt-candidates':51036, 'unrel_candidates':166368, 'unrel_gt-candidates':10308}

for cand_test in ['gt-candidates','candidates']:
#for cand_test in ['gt-candidates']:
    for data_name in datasets:

        # Load the dataset
        dset, loader = load_dataset(data_name, cand_test)

        res_scores = get_scores(loader, keys)

        res_path = osp.join(save_dir, 'scores_{}_{}_{}_{}_{}.mat'.format(data_name, \
                                                                cand_test,\
                                                                opt.test_split,\
                                                                opt.epoch_model,\
                                                                '%s'))
        for key in keys:

            if key in ['s','o']:
                if opt.use_objscoreprecomp:
                    keyname = key+'-objscoreprecomp'
                    res_file = res_path % keyname
                else:
                    res_file = res_path % key
            else:
                res_file = res_path % key

            # Sort scores by rel_id to match matlab iccv evaluation code -> argsort by rel_id
            scores = res_scores[key][:,1:]
            rel_id = res_scores[key][:,0]

            # Check there is right number of candidates
            num_cand = len(np.unique(rel_id))
            assert num_cand==num_cand_theoric[data_name + '_' + cand_test] and np.max(rel_id)==num_cand_theoric[data_name + '_' + cand_test], 'Not right number of candidates'

            idx_sort = np.argsort(rel_id)
            scores = scores[idx_sort,:]

            sio.savemat(res_file, {'scores':scores})




