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
from datasets.BaseLoader import TestSampler
from networks import models
import scipy.io as sio
import yaml
import warnings
warnings.filterwarnings("ignore")
import os
import torch.nn.functional as F
import csv
from datasets.hico_api import Hico 
from datasets.cocoa_api import Cocoa

def add_new_queries(model, new_queries, new_predicates, new_word_embeddings):
    """ Add new predicates in vocab """

    word_embeddings = np.array(model.word_embeddings)
    triplet_mass = np.array(model.triplet_mass)
    
    for count, new_predicate in enumerate(new_predicates):
        if new_predicate not in model.vocab['r'].words():
            model.vocab['r'].add_word(new_predicate, 'verb')
            model.vocab['all'].add_word(new_predicate, 'verb')
            word_embeddings = np.vstack((word_embeddings, new_word_embeddings[count,:]))

    for _, new_query in enumerate(new_queries):
        if new_query not in model.vocab['sro'].words():
            model.vocab['sro'].add_word(new_query, 'noun-verb-noun')
            triplet_mass = np.hstack((triplet_mass, np.array([0])))

    idx_sro_to   = dset_train.get_idx_between_vocab(model.vocab['sro'], model.vocab)
    idx_to_vocab = dset_train.get_idx_in_vocab(model.vocab, model.vocab['all'])
    
    model.word_embeddings = word_embeddings
    model.triplet_mass = triplet_mass
    
    model.idx_sro_to = {}
    model.idx_sro_to_numpy = {}
    for key in idx_sro_to.keys():
        model.idx_sro_to_numpy[key] = idx_sro_to[key].astype(int)
        model.idx_sro_to[key] = Variable(torch.from_numpy(idx_sro_to[key])).long()
        if opt.use_gpu:
            model.idx_sro_to[key] = model.idx_sro_to[key].cuda()

    model.idx_to_vocab = {}
    model.idx_to_vocab_numpy = {}
    for key in idx_to_vocab.keys():
        model.idx_to_vocab_numpy[key] = idx_to_vocab[key].astype(int)
        model.idx_to_vocab[key] = Variable(torch.from_numpy(idx_to_vocab[key])).long()
        if opt.use_gpu:
            model.idx_to_vocab[key] = model.idx_to_vocab[key].cuda()

    return model



""" Parsing options """

args = argparse.ArgumentParser()
parser = Parser(args)
opt = parser.make_options()


def get_detections(loader, keys, target_triplets):

    large_number = 2000000 # pre-initialize detection matrices 

    # Detections : detections[key]['all_boxes']
    detections = {}
    counts = {}
    print('Pre-init the detections and counts matrices...')

    for key in keys:
        detections[key] = {}
        counts[key] = {}

        for count,target_triplet in enumerate(target_triplets):
            
            print('Pre-init : {}/{}'.format(count,len(target_triplets)))

            detections[key][target_triplet] = np.zeros((large_number,11), dtype=np.float32)  # [im_id, subject_box, object_box, conf, cand_id]
            counts[key][target_triplet] = 0


    start_time = time.time()
    for batch_idx, batch_input in enumerate(loader):

        # Get boxes and object class
        im_ids        = batch_input['cand_info'][:,0].numpy()
        cand_ids      = batch_input['cand_info'][:,1].numpy()
        pair_objects  = batch_input['pair_objects'].numpy()
        subject_boxes = pair_objects[:,0,:4]
        sub_classes   = pair_objects[:,0,4]
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
                vis_feats[gram] = model.get_visual_features(batch_input, gram) 


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
        for triplet_id, target_triplet in enumerate(target_triplets):

            query_sub_cat, query_rel_cat, query_obj_cat = target_triplets_idx[triplet_id,:]
            query_rel_cat_modelvocab = model.vocab['r'].word2idx[dset_test.predicates.idx2word[query_rel_cat]] 

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
                scores_gram['r'] = scores['r'][idx, query_rel_cat_modelvocab].data[0]
                scores_gram['o'] = scores['o'][idx, query_obj_cat].data[0]
                scores_gram['sro'] = scores_sro[idx, triplet_id].data[0]


                for key in keys:
                    score = 1
                    grams = key.split('-')
                    for gram in grams:
                        score *= scores_gram[gram]


                    start_id = counts[key][target_triplet]
                    end_id = start_id + 1

                    # Increase size matrix
                    if end_id >= detections[key][target_triplet].shape[0]:
                        new_array = np.zeros((large_number+detections[key][target_triplet].shape[0],11))
                        new_array[:start_id,:] = np.array(detections[key][target_triplet][:start_id,:])
                        detections[key][target_triplet] = new_array

                    detections[key][target_triplet][start_id:end_id,:] = [im_id] + list(subject_box) + list(object_box) + [score] + [cand_id]

                    counts[key][target_triplet] = end_id


        #del batch_input

        if batch_idx % 100 ==0:
            print('Done [{}/{}] in {:.2f} sec'.format(batch_idx, len(loader), time.time()-start_time))
            start_time = time.time()

    for key in keys:
        for _,target_triplet in enumerate(target_triplets):
            
            detections[key][target_triplet] = detections[key][target_triplet][:counts[key][target_triplet],:]


    return detections


####################
""" Data loaders """
####################

store_ram = []
if opt.use_ram:
    if opt.use_precompobjectscore:
        store_ram.append('objectscores')
    if opt.use_precompappearance:
        store_ram.append('appearance')


data_path  = '{}/{}'.format(opt.data_path, '%s')
image_path = '{}/{}/{}'.format(opt.data_path, '%s', 'images')
cand_dir   = '{}/{}/{}'.format(opt.data_path, '%s', 'detections')

dset_train = Hico(  data_path % 'hicoforcocoa',\
                    image_path % 'hicoforcocoa', \
                    opt.train_split, \
                    cand_dir = cand_dir % 'hicoforcocoa',\
                    thresh_file = opt.thresh_file, \
                    use_gt = opt.use_gt, \
                    add_gt = opt.add_gt, \
                    train_mode = False, \
                    jittering = False, \
                    nms_thresh = opt.nms_thresh,\
                    store_ram = [],\
                    l2norm_input = opt.l2norm_input,\
                    neg_GT = opt.neg_GT)


dset_test = Cocoa(  data_path % 'cocoa',\
                    image_path % 'cocoa', \
                    'all', \
                    cand_dir = cand_dir % 'cocoa',\
                    thresh_file = opt.thresh_file, \
                    use_gt = False, \
                    add_gt = False, \
                    train_mode = False, \
                    jittering = False, \
                    nms_thresh = opt.nms_thresh,\
                    store_ram = store_ram,\
                    l2norm_input = opt.l2norm_input,\
                    neg_GT = opt.neg_GT)


dset_loader = TestSampler(dset_test,\
                use_precompappearance   = opt.use_precompappearance, \
                use_precompobjectscore  = opt.use_precompobjectscore)


loader = torch.utils.data.DataLoader(dset_loader, batch_size=8, shuffle=False, num_workers=0, collate_fn=dset_loader.collate_fn)


##################
""" Load model """
##################

logger_path = osp.join(opt.logger_dir, opt.exp_name)
opt = parser.get_opts_from_dset(opt, dset_train)

# Load model
print('Loading model')
model = models.get_model(opt)
checkpoint = torch.load(osp.join(logger_path, 'model_' + opt.epoch_model + '.pth.tar'), map_location=lambda storage, loc: storage)
model.load_state_dict(checkpoint['model'])

model.eval()


####################################
""" Get the triplets to retrieve """
####################################

subset_test = 'all' # 'all','unseen'
target_triplets = dset_test.get_triplets_subset(subset_test)


print('Computing retrieval on {} triplet queries'.format(len(target_triplets)))


###################################
""" Merge vocab HICO and COCO-a """
###################################

# Add new predicates in vocab 

new_predicates      = dset_test.predicates.words()
new_queries         = dset_test.visualphrases.words()
new_word_embeddings = dset_test.word_embeddings

model = add_new_queries(model, new_queries, new_predicates, new_word_embeddings)

# Replace embedding layer
for j in range(len(model.language_nets)):
    model.language_nets[j][0].emb = nn.Embedding(model.word_embeddings.shape[0],300)
    model.language_nets[j][0].word_embeddings = model.word_embeddings
    model.language_nets[j][0].init_weights()

# Multiple gpus
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)

if torch.cuda.is_available():
    model.cuda()

model.eval()

###########################
""" Precomp for speedup """
###########################

if opt.use_analogy:
    model.precomp_language_features()

    if opt.precomp_vp_source_embedding:
        model.precomp_source_queries()

    
# Target queries indices
queries_sro, triplet_queries_idx = model.precomp_target_queries(triplet_queries)


# Pre-compute language features in joint sro space
print('Precomputing query features in joint space with analogy...')
lang_feats_precomp_sro = model.get_language_features_analogy(queries_sro)


################
""" Evaluate """
################

keys = opt.mixture_keys.split('_') if opt.mixture_keys else ['s-r-o','s-r-o-sro', 's-sro-o']
save_dir = parser.get_res_dir(opt, 'apRetrieval_{}_{}'.format(subset_test, opt.embedding_type))

##########################
""" Get the detections """
##########################

# Save the detections by key (group all triplets together)
det_path = osp.join(save_dir, 'detections_{}_{}_{}_{}_{}.pkl'.format(opt.cand_test,\
                                                                opt.epoch_model,\
                                                                opt.data_name,\
                                                                '%s',\
                                                                '%s'))

det_file = det_path % (subset_test, 's-r-o')


print('Begin evaluation')
if not osp.exists(det_file):
    
    all_detections = get_detections(loader, keys, target_triplets)

    for key in keys:

        if opt.use_objscoreprecomp:
            keyname = key+'-objscoreprecomp'
            det_file = det_path % (subset_test, keyname)
        else:
            det_file = det_path % (subset_test, key)

        pickle.dump(all_detections[key], open(det_file,'wb'))
else:

    all_detections = {}
    for key in keys:

        if opt.use_objscoreprecomp:
            keyname = key+'-objscoreprecomp'
            det_file = det_path % (subset_test, keyname)
        else:
            det_file = det_path % (subset_test, key)

        all_detections[key] = pickle.load(open(det_file,'rb'))

#############
## Get GT ###
#############

# Could pre-compute and save to speed-up
gt, npos = dset_test.get_gt(target_triplets)


############
#### AP ####
############

unseen_triplets = dset_test.get_triplets_subset('unseen')
outofvocab_triplets = dset_test.get_triplets_subset('outofvocab')


res_path = osp.join(save_dir, 'res_apRetrieval_{}_{}_{}_{}_{}.csv'.format(opt.cand_test,\
                                                        opt.epoch_model,\
                                                        opt.data_name,\
                                                        '%s',\
                                                        '%s'))


for key in keys:

    detections = all_detections[key]

    if opt.use_objscoreprecomp:
        keyname = key+'-objscoreprecomp'
        det_file = det_path % (subset_test, keyname)
        res_file = res_path % (subset_test, keyname)
    else:
        det_file = det_path % (subset_test, key)
        res_file = res_path % (subset_test, key)

    # Compute AP
    with open(res_file, 'wb') as f:
        res_writer = csv.writer(f, delimiter=',')

        ap = np.zeros((len(target_triplets),))
        recall = np.zeros((len(target_triplets),))
        idx_unseen = np.zeros((len(target_triplets),))
        idx_outvocab = np.zeros((len(target_triplets),))

        for r,target_triplet in enumerate(target_triplets):

            print('Computing AP {}/{}'.format(r, len(target_triplets)))

            detections_triplet  = detections[target_triplet]
            gt_triplet          = gt[target_triplet]
            npos_triplet        = npos[target_triplet]
            ap[r], recall[r]    = dset_test.eval_speed(detections_triplet, gt_triplet, npos_triplet, min_overlap=0.5)

            if target_triplet in unseen_triplets:
                idx_unseen[r] = 1

            if target_triplet in outofvocab_triplets: 
                idx_outvocab[r] = 1

        # Write the mean
        ap_mean = '{:.2f}'.format(np.nanmean(ap)*100)
        recall_mean = '{:.2f}'.format(np.nanmean(recall)*100)
        res_writer.writerow(['mAP', ap_mean, recall_mean])
        res_writer.writerow(['mAP unseen', '{:.2f}'.format(np.nanmean(ap[idx_unseen.astype(bool)])*100), \
                                           '{:.2f}'.format(np.nanmean(recall[idx_unseen.astype(bool)])*100)])
        res_writer.writerow(['mAP out of vocab', '{:.2f}'.format(np.nanmean(ap[idx_outvocab.astype(bool)]*100)), \
                                                 '{:.2f}'.format(np.nanmean(recall[idx_outvocab.astype(bool)]*100))])

        for r, target_triplet in enumerate(target_triplets):
            ap_triplet = '{:.2f}'.format(ap[r]*100)
            recall_triplet = '{:.2f}'.format(recall[r]*100)
            res_writer.writerow([target_triplet, ap_triplet, recall_triplet])


