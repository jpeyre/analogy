"""
Gather results evaluation HICO-DET
"""

from __future__ import division
import __init__
import os
import pickle
import os.path as osp
from datasets.hico_api import Hico
import torch
import argparse
import csv
from utils import Parser


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


""" Key types """
keys = ['s-r-o','s-sro-o','s-r-o-sro']

""" Load the test triplets """
target_triplets = dset.get_zeroshottriplets()
subset = 'zeroshottriplet'

for key in keys:

    """ Load ap results for all triplets """
    filename_in = osp.join(opt.logger_dir, opt.exp_name, 'results_{}_{}_{}_{}_def.csv'.format(\
                                        opt.cand_test,\
                                        opt.test_split,\
                                        opt.epoch_model,\
                                        key))

    with open(filename_in) as f:
        reader = csv.DictReader(f)
        ap_results = [r for r in reader]

    """  Write csv subset of triplets """
    filename_out = osp.join(opt.logger_dir, opt.exp_name, 'results_{}_{}_{}_{}_{}_def.csv'.format(\
                                        opt.cand_test,\
                                        subset,\
                                        opt.test_split,\
                                        opt.epoch_model,\
                                        key))

    with open(filename_out, 'wb') as f:
        writer = csv.writer(f)

        mean_ap = 0
        mean_recall = 0
        for _,target_triplet in enumerate(target_triplets):

            r = dset.visualphrases.word2idx[target_triplet]
            writer.writerow([ap_results[r][key] for key in ap_results[r].keys()])

            mean_ap += float(ap_results[r]['AP'])
            mean_recall += float(ap_results[r]['REC'])

        mean_ap /= len(target_triplets)
        mean_recall /= len(target_triplets)

        writer.writerow(['mean_AP', '{:.3f}'.format(mean_ap), '{:.3f}'.format(mean_recall)])




