"""
Get detections with analogy
"""

from __future__ import division
import __init__
import tensorboard_logger 
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import time
from utils import Parser
import argparse
import os.path as osp
import cPickle as pickle
from datasets.hico_api import Hico
from datasets.BaseLoader import TestSampler
from networks import models
import scipy.io as sio
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")
import os


"""
Parsing options
"""

args = argparse.ArgumentParser()
parser = Parser(args)
opt = parser.make_options()
print(opt)


"""
Get detections
"""

def get_detections(loader, triplet_queries):

    large_number = 100 # pre-initialize detection matrices 

    # Detections : detections[key]['all_boxes']
    detections = {}
    counts = {}
    print('Pre-init the detections and counts matrices...')
    for count,target_triplet in enumerate(triplet_queries):

        print('Pre-init : {}/{}'.format(count,len(triplet_queries)))
    
        detections[target_triplet] = {}
        counts[target_triplet] = {}

        for key in keys:
            detections[target_triplet][key] = {}
            counts[target_triplet][key] = {}

            # By image (to speed up)
            for im_id in images:
                detections[target_triplet][key][im_id] = np.zeros((large_number,10), dtype=np.float32)  # [subject_box, object_box, conf, cand_id, im_id]
                counts[target_triplet][key][im_id] = 0


    start_time = time.time()
    print('Loop over batch...')
    for batch_idx, batch_input in enumerate(loader):

        # Get boxes and object class
        im_ids        = batch_input['cand_info'][:,0].numpy()
        cand_ids      = batch_input['cand_info'][:,1].numpy()
        pair_objects  = batch_input['pair_objects'].numpy()
        subject_boxes = pair_objects[:,0,:4]
        object_boxes  = pair_objects[:,1,:4]
        obj_classes   = pair_objects[:,1,4]

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


        """ Get the scores of compositional model : s-r-o / r / s / o """
        scores = model.get_scores(batch_input)
        if isinstance(scores,tuple):
            scores = scores[0]

        """ Attach object scores : else, object scores by object branches (to try actually) """
        if opt.use_objscoreprecomp:
            scores = model.attach_objectscores_detectors(batch_input, scores)


        scores = model.form_factors(scores, keys) # form scores of interest


        """ Compute sro scores """
        scores_sro = torch.matmul(vis_feats['sro'], lang_feats_precomp_sro.transpose(0,1))
        scores_sro = F.sigmoid(scores_sro)


        """ Get the scores of visual phrase branch for different source triplet """
        # Loop over the target triplet
        for triplet_id, target_triplet in enumerate(triplet_queries):

            query_sub_cat, query_rel_cat, query_obj_cat = triplet_queries_idx[triplet_id,:]


            # Find examples in batch matching query_obj_cat
            idx_cat = np.where(obj_classes == query_obj_cat)[0]


            # Loop to arange over images
            for _,idx in enumerate(idx_cat):

                im_id       = im_ids[idx]
                cand_id     = cand_ids[idx]
                subject_box = subject_boxes[idx,:]
                object_box  = object_boxes[idx,:]

                # Scores
                scores_gram = {}
                scores_gram['s'] = scores['s'][idx, query_sub_cat].data[0]
                scores_gram['r'] = scores['r'][idx, query_rel_cat].data[0]
                scores_gram['o'] = scores['o'][idx, query_obj_cat].data[0]
                scores_gram['sro'] = scores_sro[idx, triplet_id].data[0]


                for key in keys:

                    score = 1
                    grams = key.split('-')
                    for gram in grams:
                        score *= scores_gram[gram]


                    start_id = counts[target_triplet][key][im_id]
                    end_id = start_id + 1

                    # Increase size matrix
                    if end_id >= detections[target_triplet][key][im_id].shape[0]:
                        new_array = np.zeros((100+detections[target_triplet][key][im_id].shape[0],10))
                        new_array[:start_id,:] = np.array(detections[target_triplet][key][im_id][:start_id,:])
                        detections[target_triplet][key][im_id] = new_array

                    detections[target_triplet][key][im_id][start_id:end_id,:] = list(subject_box) + list(object_box) + [score] + [cand_id]

                    counts[target_triplet][key][im_id] = end_id
            

        if batch_idx % 100 ==0:
            print('Done [{}/{}] in {:.2f} sec'.format(batch_idx, len(loader), time.time()-start_time))
            start_time = time.time()


    for _,target_triplet in enumerate(triplet_queries):
            for key in keys:
                for im_id in images:
                    detections[target_triplet][key][im_id] = detections[target_triplet][key][im_id][:counts[target_triplet][key][im_id],:]

    
    return detections


####################
""" Data loaders """
####################

store_ram = []
store_ram.append('objectscores') if opt.use_ram and opt.use_precompobjectscore else None
store_ram.append('appearance') if opt.use_ram and opt.use_precompappearance else None

data_path  = '{}/{}'.format(opt.data_path, opt.data_name)
image_path = '{}/{}/{}'.format(opt.data_path, opt.data_name, 'images')
cand_dir   = '{}/{}/{}'.format(opt.data_path, opt.data_name, 'detections')

dset = Hico(data_path, \
            image_path, \
            opt.test_split, \
            cand_dir = cand_dir,\
            thresh_file = opt.thresh_file,\
            use_gt = False, \
            train_mode = False, \
            jittering = False,\
            nms_thresh = opt.nms_thresh, \
            store_ram = store_ram, \
            l2norm_input = opt.l2norm_input)


dset_loader = TestSampler(dset, \
                    use_image = opt.use_image, \
                    use_precompappearance = opt.use_precompappearance, \
                    use_precompobjectscore = opt.use_precompobjectscore)


loader = torch.utils.data.DataLoader(dset_loader, batch_size=8, shuffle=False, num_workers=0, collate_fn=dset_loader.collate_fn)


##################
""" Load model """
##################

# Logger path
logger_path = osp.join(opt.logger_dir, opt.exp_name)


save_dir = parser.get_res_dir(opt, 'detections_' + opt.embedding_type)
opt = parser.get_opts_from_dset(opt, dset) # additional opts from dset


# Model
model = models.get_model(opt)

checkpoint = torch.load(osp.join(logger_path, 'model_' + opt.epoch_model + '.pth.tar'))
model.load_state_dict(checkpoint['model'])
model.eval()

# Multiple gpus
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)

if torch.cuda.is_available():
    model.cuda()

######################
""" Query triplets """
######################

if opt.train_split=='trainval' or opt.train_split=='train':
    triplet_queries = dset.visualphrases.words()
else:
    triplet_queries = dset.get_zeroshottriplets()



###############
""" Analogy """
###############

if opt.use_analogy:
    model.precomp_language_features()

    if opt.precomp_vp_source_embedding:
        model.precomp_source_queries()


# Target queries indices
queries_sro, triplet_queries_idx = model.precomp_target_queries(triplet_queries)


# Pre-compute language features in joint sro space
print('Precomputing query features in joint space with analogy...')
lang_feats_precomp_sro = model.get_language_features_analogy(queries_sro)



##########################################
""" Get detections for target triplets """
##########################################


""" Get images of interest """
images = np.array(dset.image_ids, dtype=int)


""" Get keys of interest """
keys = opt.mixture_keys.split('_') if opt.mixture_keys else ['s-r-o-sro','s-sro-o','s-r-o']


""" Get all detections for the target triplets """
all_detections = get_detections(loader, triplet_queries)


# We save in separate .mat files 
det_path = osp.join(save_dir,'detections_{}_{}_{}_{}_{}.mat'.format(opt.cand_test,\
                                                        opt.test_split,\
                                                        opt.epoch_model,\
                                                        '%s',\
                                                        '%s'))


for _,target_triplet in enumerate(triplet_queries):

    for key in keys:

        detections_aggregsource = all_detections[target_triplet][key]

        if opt.use_objscoreprecomp:
            keyname = key+'-objscoreprecomp'
            det_file = det_path %(target_triplet, keyname)
        else:
            det_file = det_path %(target_triplet, key)


        # Sort by image for eval code in matlab -> specific format
        detections = {}
        detections['all_boxes'] = []

        for im_id in images:
            detections['all_boxes'].append(detections_aggregsource[im_id]) 

        # For speed-up: only save detection of the target triplet
        sio.savemat(det_file, {'all_boxes':detections['all_boxes']})


