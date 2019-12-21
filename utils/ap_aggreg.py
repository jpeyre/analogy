"""
Utils for evaluation on HICO-DET. This script helps you gather and group the results from official HICO-DET evaluation code. 
"""

from __future__ import division
import __init__
import os
import os.path as osp
import torch
import argparse
import csv
from datasets.hico_api import Hico
from utils import Parser
from networks import models
from torch.autograd import Variable

import numpy as np

""" Parsing options """
args = argparse.ArgumentParser()
parser = Parser(args)
opt = parser.make_options()


""" Load dataset """
data_path  = '{}/{}'.format(opt.data_path, 'hico')
image_path = '{}/{}/{}'.format(opt.data_path, 'hico', 'images')
cand_dir   = '{}/{}/{}'.format(opt.data_path, 'hico', 'detections')

dset = Hico(data_path, \
            image_path, \
            opt.test_split, \
            cand_dir=cand_dir,\
            thresh_file=opt.thresh_file, \
            add_gt=False, \
            train_mode=False, \
            jittering=False, \
            nms_thresh=opt.nms_thresh)


""" Load the test triplets """
target_triplets = dset.get_zeroshottriplets() # uncomment to eval zeroshot triplets
#target_triplets = dset.visualphrases.words() # uncomment to eval all triplets


""" Keys to analyze """
keys = ['s-sro-o','s-r-o-sro']


""" Aggregate csv result files (from official HICO eval code) """
# Logger path
logger_path = osp.join(opt.logger_dir, opt.exp_name)

detection_path = parser.get_res_dir(opt, 'detections_' + opt.embedding_type)
res_path       = parser.get_res_dir(opt, 'res_' + opt.embedding_type)


for key in keys:

    """ File out : 1 file for AP results : group all zeroshot triplets AP """
    filename_out = osp.join(res_path, 'results_{}_{}_{}_{}.csv'.format(\
                                        opt.cand_test,\
                                        opt.test_split,\
                                        opt.epoch_model,\
                                        key))

    with open(filename_out, 'wb') as f:
        writer = csv.writer(f)

        ap_triplets = []

        for _,target_triplet in enumerate(target_triplets):

            filename_in = osp.join(detection_path, 'results_{}_{}_{}_{}_{}_def.csv'.format(\
                                                opt.cand_test,\
                                                opt.test_split,\
                                                opt.epoch_model,\
                                                target_triplet,\
                                                key))

            with open(filename_in) as h:
                reader = csv.DictReader(h)
                ap_results = [r for r in reader]
                ap = float(ap_results[0]['AP'])
                writer.writerow([target_triplet, ap])
                ap_triplets.append(ap)

        # Write mAP
        writer.writerow(['mAP', np.nanmean(ap_triplets)])


"""
Get the similarities used for predictions : in detection_dir. 1 file for each target triplet (more readable)
"""

if not opt.embedding_type=='target':


    occurrences_train = dset.get_occurrences_precomp(opt.train_split.split('_')[0])
    opt.occurrences   = dset.get_occurrences_precomp(opt.train_split.split('_')[0])


    opt = parser.get_opts_from_dset(opt, dset) # additional opts from dset


    model = models.get_model(opt)
    checkpoint = torch.load(osp.join(logger_path, 'model_' + opt.epoch_model + '.pth.tar'), map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    if torch.cuda.is_available():
        model.cuda()


    """ Pre-compute language feats """

    if opt.use_analogy:
        model.precomp_language_features()

        if opt.precomp_vp_source_embedding:
            model.precomp_source_queries()


    queries_sro, _ = model.precomp_target_queries(target_triplets)


    """ Get similarities with source """

    for count,target_triplet in enumerate(target_triplets):


        if opt.use_analogy:

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


        filename_out = osp.join(detection_path, 'similarities_{}_{}_{}_{}_{}.csv'.format(\
                                            opt.cand_test,\
                                            opt.test_split,\
                                            opt.epoch_model,\
                                            target_triplet,\
                                            key))

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
                    source_triplet = '-'.join([ dset.vocab_grams['all'].idx2word[sub_cat_source],\
                                            dset.vocab_grams['all'].idx2word[rel_cat_source],\
                                            dset.vocab_grams['all'].idx2word[obj_cat_source]])


                else:
                    source_triplet = '-'.join([ dset.vocab_grams['s'].idx2word[sub_cat_source],\
                                            dset.vocab_grams['r'].idx2word[rel_cat_source],\
                                            dset.vocab_grams['o'].idx2word[obj_cat_source]])


                source_sim = '{:.3f}'.format(similarities[s])


                # Occurrence
                occ_train = occurrences_train[source_triplet]

                # Csv
                writer.writerow([source_triplet, source_sim, occ_train]) 



