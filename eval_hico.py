"""
Compute scores for all candidate pairs
"""

from __future__ import division
import __init__
import tensorboard_logger 
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import time
import argparse
import os.path as osp
import cPickle as pickle
from utils import Parser
from datasets.hico_api import Hico
from datasets.BaseLoader import TestSampler
from networks import models
import scipy.io as sio
import yaml
import warnings
warnings.filterwarnings("ignore")
import os


"""
Parsing options
"""

args = argparse.ArgumentParser()
parser = Parser(args)
opt = parser.make_options()


"""
Get detections
"""

def get_detections(loader, keys):

    large_number = 2000000 # pre-initialize detection matrices 

    # Detections : detections[key]['all_boxes']
    detections = {}
    counts = {}
    for key in keys:
        detections[key] = [np.zeros((large_number,11), dtype=np.float32) for r in range(len(dset.visualphrases))]  # [subject_box, object_box, conf, cand_id, im_id]
        counts[key] = [0 for r in range(len(dset.visualphrases))]


    start_time = time.time()
    for batch_idx, batch_input in enumerate(loader):

        # Get boxes and object class
        im_id         = batch_input['cand_info'][:,0].numpy()
        cand_id       = batch_input['cand_info'][:,1].numpy()
        pair_objects  = batch_input['pair_objects'].numpy()
        subject_boxes = pair_objects[:,0,:4]
        object_boxes  = pair_objects[:,1,:4]
        obj_classes   = pair_objects[:,1,4]


        scores_batch = {key:[] for key in keys}

        for inpt in batch_input.keys():
            if opt.use_gpu:
                batch_input[inpt] = Variable(batch_input[inpt].cuda())
            else:
                batch_input[inpt] = Variable(batch_input[inpt])


        # Forward scores
        scores = model.get_scores(batch_input) 

        if isinstance(scores,tuple):
            scores = scores[0]

        """ Attach object scores : else, object scores by object branches (to try actually) """
        if opt.use_objscoreprecomp:
            scores = model.attach_objectscores_detectors(batch_input, scores)

        # Combine scores
        scores = model.form_factors(scores, keys)

        for key in keys:
            scores_batch[key] = scores[key].data.cpu().numpy()  
        
        # Get the detections for each relation: visual phrase
        for r in range(len(dset.visualphrases)):

            relation = dset.visualphrases.idx2word[r]

            # Get object / predicate
            [_, _, obj_name] = relation.split('-')
            obj_cat = dset.classes.wordpos2idx[obj_name + '_noun']

            # Filter detections by object category 
            idx = np.where(obj_classes==obj_cat)[0]

            # Get predictions for each branch
            for key in keys:

                if len(idx)>0:

                    scores = np.array(scores_batch[key][idx,r])

                    start_id = counts[key][r]
                    end_id = start_id + scores.shape[0]

                    # Increase size matrix
                    if end_id >= detections[key][r].shape[0]:
                        new_array = np.zeros((1000000+detections[key][r].shape[0],11))
                        new_array[:start_id,:] = np.array(detections[key][r][:start_id,:])
                        detections[key][r] = new_array

                    detections[key][r][start_id:end_id,:] = np.hstack((subject_boxes[idx,:],\
                                                                             object_boxes[idx,:],\
                                                                             scores[:,None],\
                                                                             cand_id[idx,None],\
                                                                             im_id[idx,None]))

                    counts[key][r] = end_id
            

        if batch_idx % 100 ==0:
            print('Done [{}/{}] in {:.2f} sec'.format(batch_idx, len(loader), time.time()-start_time))
            start_time = time.time()


    for key in keys:
        for r in range(len(dset.visualphrases)):
            detections[key][r] = detections[key][r][:counts[key][r],:]

    
    return detections



####################
""" Data loaders """
####################

store_ram = []
store_ram.append('objectscores') if opt.use_ram and opt.use_precompobjectscore else None
store_ram.append('appearance') if opt.use_ram and opt.use_precompappearance else None

data_path  = '{}/{}'.format(opt.data_path, 'hico')
image_path = '{}/{}/{}'.format(opt.data_path, 'hico', 'images')
cand_dir   = '{}/{}/{}'.format(opt.data_path, 'hico', 'detections')

dset = Hico(data_path, \
            image_path, \
            opt.test_split, \
            cand_dir = cand_dir,\
            thresh_file = opt.thresh_file, \
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


opt = parser.get_opts_from_dset(opt, dset) # additional opts from dset


model = models.get_model(opt)
checkpoint = torch.load(osp.join(logger_path, 'model_' + opt.epoch_model + '.pth.tar'), map_location=lambda storage, loc: storage)
model.load_state_dict(checkpoint['model'])
model.eval()

# Multiple gpus
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)

if torch.cuda.is_available():
    model.cuda()


###############################
""" Get detections and save """
###############################


det_path = osp.join(logger_path,'detections_{}_{}_{}_{}.mat'.format(opt.cand_test,\
                                                        opt.test_split,\
                                                        opt.epoch_model,\
                                                        '%s'))


keys = opt.mixture_keys.split('_') if opt.mixture_keys else ['s-r-o','s-r-o-sro'] 


detections = {}
if not osp.exists(det_path %keys[0]):

    images = np.array(dset.image_ids, dtype=int)
    all_detections = get_detections(loader, keys)
    for key in keys:

        if opt.use_objscoreprecomp:
            keyname = key + '-objscoreprecomp'
            det_file = det_path % keyname
        else:
            det_file = det_path % key 

        # Sort by image to match official eval code
        detections[key] = {}
        detections[key]['all_boxes'] = [[] for r in range(len(dset.visualphrases))]

        for r in range(len(dset.visualphrases)):
            for im_id in images:
                idx = np.where(all_detections[key][r][:,10]==im_id)[0]
                detections[key]['all_boxes'][r].append(all_detections[key][r][idx,:10]) 

        # Save in matlab to use official eval code
        sio.savemat(det_file, detections[key])



