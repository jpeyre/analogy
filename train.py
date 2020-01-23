from __future__ import division
import __init__
import tensorboard_logger
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import time
from utils import AverageMeter, Tb_logger, Parser
import argparse
import os.path as osp
import os
from datasets.BaseLoader import TrainSampler
from networks import models
import yaml
import warnings
warnings.filterwarnings("ignore")


"""
Parsing options
"""

args = argparse.ArgumentParser()
parser = Parser(args)
opt = parser.make_options()
print(opt)


"""
Train / val
"""

def train(epoch, split):

    batch_time = 0
    train_loss = {}
    train_recall = {}
    train_precision = {}

    loader = loaders[split]
    model.train()

    start_time = time.time()
    start = time.time()

    for batch_idx, batch_input in enumerate(loader):

        for key in batch_input.keys():
            if opt.use_gpu:
                batch_input[key] = Variable(batch_input[key].cuda())
            else:
                batch_input[key] = Variable(batch_input[key])

        # Train
        loss, tp_class, fp_class, num_pos_class = model.train_(batch_input)


        batch_time += time.time() - start
        start = time.time()

        # True pos/false pos per branch
        for gram in tp_class.keys():

            recall = np.nanmean(tp_class[gram].numpy()/num_pos_class[gram].numpy())
            precision = np.nanmean(tp_class[gram].numpy() / (tp_class[gram].numpy() + fp_class[gram].numpy()))

            if gram not in train_recall.keys():
                train_recall[gram] = AverageMeter()

            if gram not in train_precision.keys():
                train_precision[gram] = AverageMeter()

            if gram not in train_loss.keys():
                train_loss[gram] = AverageMeter()


            train_recall[gram].update(recall, n=batch_input['pair_objects'].size(0))
            train_precision[gram].update(precision, n=batch_input['pair_objects'].size(0))
            train_loss[gram].update(loss[gram].data[0], n=batch_input['pair_objects'].size(0))

        # Loss reg
        if opt.use_analogy:
            if 'reg' not in train_loss.keys():
                train_loss['reg'] = AverageMeter()
            train_loss['reg'].update(loss['reg'].data[0], n=batch_input['pair_objects'].size(0))


        learning_rate = model.optimizer.param_groups[0]['lr']

        if batch_idx % 100 ==0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tDone in: {:.2f} sec'.format(epoch, batch_idx, len(loader), 100. * batch_idx / len(loader), sum(loss.values()).data[0], (time.time()-start_time)))
            start_time = time.time()
        
        
        # Record logs in tensorboard
        if model.ite % 500 ==0:

            batch_time /= 500
           
            total_train_loss = 0
            if opt.use_analogy:
                total_train_loss = train_loss['sro'].avg + opt.lambda_reg*train_loss['reg'].avg
            else:
                for _, val in train_loss.iteritems():
                    total_train_loss += val.avg

            # Register in logger 
            tb_logger[split].log_value('epoch', epoch, model.ite)
            tb_logger[split].log_value('loss', total_train_loss, model.ite)
            tb_logger[split].log_value('batch_time', batch_time, model.ite)
            tb_logger[split].log_value('learning_rate', learning_rate, model.ite)
            tb_logger[split].log_value('weight_decay', opt.weight_decay, model.ite)

            for gram in tp_class.keys():
                tb_logger[split].log_value(gram+'_loss', train_loss[gram].avg, model.ite)
                tb_logger[split].log_value(gram+'_mean_recall', 100.*train_recall[gram].avg, model.ite)
                tb_logger[split].log_value(gram+'_mean_precision', 100.*train_precision[gram].avg, model.ite)

            # Analogy loss
            if opt.use_analogy:
                tb_logger[split].log_value('loss_reg', train_loss['reg'].avg, model.ite)

            batch_time = 0

        model.ite += 1
        
    for gram in tp_class.keys():
        train_loss[gram].reset()
    if opt.use_analogy:
        train_loss['reg'].reset()




def evaluate(epoch, split):

    model.eval()

    batch_time = 0
    test_loss = {}
    test_recall = {}
    test_precision = {}


    loader = loaders[split]
    start = time.time()

    for batch_idx, batch_input in enumerate(loader):

        for key in batch_input.keys():
            if opt.use_gpu:
                batch_input[key] = Variable(batch_input[key].cuda())
            else:
                batch_input[key] = Variable(batch_input[key])

        # Eval
        loss, tp_class, fp_class, num_pos_class = model.val_(batch_input)

        batch_time += time.time() - start
        start = time.time()

        # Performance per gram
        for gram in tp_class.keys():

            recall = np.nanmean(tp_class[gram].numpy()/num_pos_class[gram].numpy())
            precision = np.nanmean(tp_class[gram].numpy() / (tp_class[gram].numpy() + fp_class[gram].numpy()))

            if gram not in test_recall.keys():
                test_recall[gram] = AverageMeter()

            if gram not in test_precision.keys():
                test_precision[gram] = AverageMeter()

            if gram not in test_loss.keys():
                test_loss[gram] = AverageMeter()

            test_recall[gram].update(recall, n=batch_input['pair_objects'].size(0))
            test_precision[gram].update(precision, n=batch_input['pair_objects'].size(0))
            test_loss[gram].update(loss[gram].data[0], n=batch_input['pair_objects'].size(0))

        # Loss analogy
        if opt.use_analogy:
            if 'reg' not in test_loss.keys():
                test_loss['reg'] = AverageMeter()
            test_loss['reg'].update(loss['reg'].data[0], n=batch_input['pair_objects'].size(0))


    # Save total loss on test
    total_test_loss = 0
    if opt.use_analogy:
        total_test_loss = test_loss['sro'].avg + opt.lambda_reg*test_loss['reg'].avg
    else:
        for _, val in test_loss.iteritems():
            total_test_loss += val.avg

    tb_logger[split].log_value('epoch', epoch, model.ite)
    tb_logger[split].log_value('loss', total_test_loss, model.ite)
    tb_logger[split].log_value('batch_time', batch_time/len(loader), model.ite)

    # Total performance per gram
    recall_gram = {}
    loss_gram = {}
    precision_gram = {}
    recall_gram = {}

    for gram in tp_class.keys():

        tb_logger[split].log_value(gram+'_loss', test_loss[gram].avg, model.ite)
        tb_logger[split].log_value(gram+'_mean_recall', 100.*test_recall[gram].avg, model.ite)
        tb_logger[split].log_value(gram+'_mean_precision', 100.*test_precision[gram].avg, model.ite)
        recall_gram[gram]    = test_recall[gram]
        precision_gram[gram] = test_precision[gram]
        loss_gram[gram]      = test_loss[gram].avg

    print('{} set: Average loss: {:.4f}, Recall: ({:.0f}%)'.format(split, sum(loss_gram.values()), \
                                    100. * np.mean(map((lambda x:x.avg), test_recall.values()))))


    for gram in tp_class.keys():
        test_loss[gram].reset()
    if opt.use_analogy:
        test_loss['reg'].reset()


    return loss_gram, precision_gram, recall_gram


#####################
""" Define logger """
#####################

splits = [opt.train_split, opt.test_split]


# Init logger
log = Tb_logger()
logger_path = osp.join(opt.logger_dir, opt.exp_name)


if osp.exists(logger_path):
    answer = raw_input("Experiment directory %s already exists. Continue: yes/no?" %logger_path)
    assert answer=='yes', 'Please speficy another experiment directory with exp_name option'


tb_logger = log.init_logger(logger_path, splits)


# Write options in directory
parser.write_opts_dir(opt, logger_path)


####################
""" Data loaders """
####################


store_ram = []
store_ram.append('objectscores') if opt.use_ram and opt.use_precompobjectscore else None
store_ram.append('appearance') if opt.use_ram and opt.use_precompappearance else None


if opt.data_name in ['hico','hicoforcocoa']:
    from datasets.hico_api import Hico as Dataset
elif opt.data_name=='vrd':
    from datasets.vrd_api import Vrd as Dataset
elif opt.data_name=='cocoa':
    from datasets.cocoa_api import Cocoa as Dataset


loaders = {}

data_path  = '{}/{}'.format(opt.data_path, opt.data_name)
image_path = '{}/{}/{}'.format(opt.data_path, opt.data_name, 'images')
cand_dir   = '{}/{}/{}'.format(opt.data_path, opt.data_name, 'detections')


# Train split
dset = Dataset( data_path, \
                image_path, \
                opt.train_split, \
                cand_dir = cand_dir,\
                thresh_file = opt.thresh_file, \
                use_gt = opt.use_gt, \
                add_gt = opt.add_gt, \
                train_mode = True, \
                jittering = opt.use_jittering, \
                store_ram = store_ram, \
                l2norm_input = opt.l2norm_input, \
                neg_GT = opt.neg_GT)


dset_loader = TrainSampler( dset, sampler_name      = opt.sampler, \
                            num_negatives           = opt.num_negatives, \
                            use_image               = opt.use_image, \
                            use_precompappearance   = opt.use_precompappearance, \
                            use_precompobjectscore  = opt.use_precompobjectscore)

loaders[opt.train_split] = torch.utils.data.DataLoader(dset_loader, \
                                                        batch_size = opt.batch_size, \
                                                        shuffle = True, \
                                                        num_workers = opt.num_workers, \
                                                        collate_fn = dset_loader.collate_fn)



# Test split
dset = Dataset( data_path, \
                image_path, \
                opt.test_split, \
                cand_dir = cand_dir,\
                thresh_file = opt.thresh_file, \
                use_gt = opt.use_gt, \
                add_gt = opt.add_gt, \
                train_mode = True, \
                jittering = False, \
                store_ram = store_ram, \
                l2norm_input = opt.l2norm_input, \
                neg_GT = opt.neg_GT)

dset_loader = TrainSampler(dset, sampler_name       = opt.sampler,\
                            num_negatives           = opt.num_negatives, \
                            use_image               = opt.use_image, \
                            use_precompappearance   = opt.use_precompappearance, \
                            use_precompobjectscore  = opt.use_precompobjectscore)

loaders[opt.test_split] = torch.utils.data.DataLoader(dset_loader, \
                                                        batch_size = opt.batch_size, \
                                                        shuffle = False, \
                                                        num_workers = opt.num_workers, \
                                                        collate_fn = dset_loader.collate_fn)

####################
""" Define model """
####################

# Get all options
opt = parser.get_opts_from_dset(opt, dset) # additional options from dataset 


# Define model
model = models.get_model(opt)

if torch.cuda.is_available():
    model.cuda()


# Load pre-trained model
if opt.pretrained_model:
    assert opt.start_epoch, 'Indicate epoch you start from'
    if opt.start_epoch:
        checkpoint = torch.load(opt.pretrained_model, map_location=lambda storage, loc: storage)
        model.load_pretrained_weights(checkpoint['model']) 


################
""" Speed-up """
################

model.eval()

if opt.use_analogy:
    
    model.precomp_language_features() # pre-compute unigram emb
    model.precomp_sim_tables() # pre-compute similarity tables for speed-up


###########
""" Run """
###########

model.train()

print('Train classifier')
best_recall = 0
for epoch in range(opt.num_epochs):
    epoch_effective = epoch + opt.start_epoch + 1

    # Train
    model.adjust_learning_rate(opt, epoch)
    train(epoch, opt.train_split)

    # Val
    loss_test, precision_test, recall_test = evaluate(epoch, opt.test_split)
    
    if epoch_effective%opt.save_epoch==0:
        state = {
                'epoch':epoch_effective,
                'model':model.state_dict(),
                'loss':loss_test,
                'precision':precision_test,
                'recall':recall_test,
                }
        torch.save(state, osp.join(logger_path, 'model_' + 'epoch' + str(epoch_effective) + '.pth.tar'))

    if recall_test > best_recall:
        state = {
                'epoch':epoch_effective,
                'model':model.state_dict(),
                'min_loss':loss_test,
                'precision':precision_test,
                'recall':recall_test,
                }
        torch.save(state, osp.join(logger_path, 'model_best.pth.tar'))
        best_recall = recall_test
    

