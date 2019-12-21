from __future__ import division
import os.path as osp
import scipy.io as sio
import numpy as np
import cv2
import scipy.misc
from datasets.utils import multilabel_transform, get_overlap, filter_small_boxes, Vocabulary
from scipy.sparse import lil_matrix
import scipy.sparse as sparse
import cPickle as pickle
import matplotlib
import copy
import time
import numbers
import csv
from datasets.Dataset import BaseDataset

class Vrd(BaseDataset):

    def __init__(self, data_dir, image_dir, split, cand_dir='', cand_test='candidates', thresh_file=None, use_gt=False, add_gt=True, train_mode=True, jittering=False, filter_images=True, nms_thresh=0.5, store_ram=[], l2norm_input=False, neg_GT=False):
        super(Vrd, self).__init__()

        self.data_name    = 'vrd' 
        self.data_dir     = data_dir
        self.split        = split
        self.data_dir     = data_dir
        self.image_dir    = image_dir
        self.cand_test    = cand_test
        self.cand_dir     = cand_dir
        self.use_gt       = use_gt
        self.add_gt       = add_gt
        self.thresh_file  = thresh_file
        self.jittering    = jittering
        self.nms_thresh   = nms_thresh
        self.store_ram    = store_ram
        self.l2norm_input = l2norm_input

        self.d_appearance = 4096


        assert len(self.store_ram)==0, 'store in RAM not yet implemented for VRD'

        self.train_split_zeroshot = ['trainval_unrel']

        if split in self.train_split_zeroshot:
            self.image_ids = np.loadtxt(osp.join(data_dir, 'splits', self.split.split('_')[0] + '.ids'), dtype=int) # restrict set of images to validation images
        else: 
            self.image_ids = np.loadtxt(osp.join(data_dir, 'splits', self.split + '.ids'), dtype=int) # restrict set of images to validation images

        if self.split in ['train', 'val', 'trainval'] or split in self.train_split_zeroshot:
            image_filenames = sio.loadmat(osp.join(self.data_dir, 'image_filenames_' + 'train' + '.mat'), squeeze_me=True)
        else:
            image_filenames = sio.loadmat(osp.join(self.data_dir, 'image_filenames_' + 'test' + '.mat'), squeeze_me=True)    
        self.image_filenames = image_filenames['image_filenames']

        # Vocabulary
        self.classes = self.get_vocab_objects()
        self.num_classes = len(self.classes)
        self.predicates = self.get_vocab_predicates()
        self.num_predicates = len(self.predicates)

        # Vocab of visualphrases : pre-computed
        self.visualphrases = pickle.load(open(osp.join(self.data_dir, 'visualphrases_trainval.pkl'),'rb'))
        self.num_visualphrases = len(self.visualphrases)
        self.subjectpredicates = self.get_vocab_subjectpredicates(self.visualphrases)
        self.objectpredicates = self.get_vocab_objectpredicates(self.visualphrases)


        # Build or load database
        if self.split in ['trainval','train','val']:
            db_name = 'db_' + self.split + '.pkl'
        elif self.split in self.train_split_zeroshot:
            db_name = 'db_' + self.split.split('_')[0] + '.pkl'
        elif self.split=='test':
            db_name = 'db_' + self.split + '_' + self.cand_test +'.pkl'

        if osp.exists(osp.join(self.data_dir, db_name)):
            self.db = pickle.load(open(osp.join(self.data_dir, db_name),'rb'))
        else:
            self.db = self._build_db()
            pickle.dump(self.db, open(osp.join(self.data_dir, db_name),'wb'))
        

        # Object scores
        if self.split in ['train', 'val', 'trainval'] or split in self.train_split_zeroshot:
            self.objectscores = sio.loadmat(osp.join(self.data_dir, 'train', 'candidates', 'objectscores.mat'), squeeze_me=True)
        else:
            self.objectscores = sio.loadmat(osp.join(self.data_dir, 'test', 'candidates', 'objectscores.mat'), squeeze_me=True)
        self.objectscores = self.objectscores['scores']


        # Zeroshot triplets
        self.zeroshot_triplets = self.get_zeroshot_triplets()

        # Vocab wrapper
        self.vocab = self.build_vocab(self.classes, self.predicates)
        pickle.dump(self.vocab.idx2word.values(), open(osp.join(self.data_dir, 'vocab' + '.pkl'), 'wb'))

        # TMP: maintain a vocabulary of all possible triplets
        self.vocab_all_triplets = self.get_vocab_all_triplets()

        
        self.vocab_grams = {'s':self.classes,
                            'o':self.classes,
                            'r':self.predicates,
                            'sr':self.subjectpredicates,
                            'ro':self.objectpredicates,
                            'sro':self.visualphrases,
                            'all':self.vocab, 
                            'all_triplets':self.vocab_all_triplets}

        self.idx_sro_to = self.get_idx_between_vocab(self.vocab_grams['sro'], self.vocab_grams)
        self.idx_to_vocab = self.get_idx_in_vocab(self.vocab_grams, self.vocab_grams['all'])

        if osp.exists(osp.join(self.data_dir, 'idx_alltriplets_to.pkl')):
            self.idx_alltriplets_to = pickle.load(open(osp.join(self.data_dir, 'idx_alltriplets_to.pkl'),'rb'))
        else:
            self.idx_alltriplets_to = self.get_idx_between_vocab(self.vocab_grams['all_triplets'], self.vocab_grams)  
            pickle.dump(self.idx_alltriplets_to, open(osp.join(self.data_dir, 'idx_alltriplets_to.pkl'),'wb'))


        self.word_embeddings = pickle.load(open(osp.join(self.data_dir, 'pretrained_embeddings_w2v.pkl'), 'rb'))


        if self.l2norm_input:
            if (np.linalg.norm(self.word_embeddings,axis=1)==0).any():
                raise Exception('At least one word embedding vector is 0 (would cause nan after normalization)')
            self.word_embeddings = self.word_embeddings / np.linalg.norm(self.word_embeddings,axis=1)[:,None]



        # Load candidates for training. For HICO we precompute and save training candidates as this step can take some time
        if train_mode:
            if osp.exists(osp.join(self.data_dir, 'cand_positives_' + split + '.pkl')):
                print('Loading candidatest from {}'.format(osp.join(self.data_dir, 'cand_positives_' + split + '.pkl')))
                self.cand_positives = pickle.load(open(osp.join(self.data_dir, 'cand_positives_' + split + '.pkl'),'rb'))
                self.cand_negatives = pickle.load(open(osp.join(self.data_dir, 'cand_negatives_' + split + '.pkl'),'rb'))

            else:
                self.cand_positives, self.cand_negatives = self.get_training_candidates(use_gt=self.use_gt, add_gt=self.add_gt)
                pickle.dump(self.cand_positives, open(osp.join(self.data_dir, 'cand_positives_' + split + '.pkl'), 'wb'))
                pickle.dump(self.cand_negatives, open(osp.join(self.data_dir, 'cand_negatives_' + split + '.pkl'), 'wb'))

        else:
            self.candidates = self.get_test_candidates(use_gt=self.use_gt, thresh_file=self.thresh_file)



        """
        Speed-up 1 : pre-load in RAM (TODO: put in dataset object)
        """
        # Pre-load images in RAM
        if len(self.store_ram)>0:
            self.data_ram = {}
            for key in self.store_ram:
                self.data_ram[key] = {}
                print('Loading {} in RAM...'.format(key))
                for im_id in self.image_ids:
                    self.data_ram[key][im_id] = self.load_data_ram(im_id, key)

        """
        Speed-up 2 : pre-compute the np.where(cand_cat==obj_cat) in dset.cand_negatives
        """
        if train_mode:
            cand_cat = self.cand_negatives[:,3]
            self.idx_match_object_candneg = {}
            for obj_cat in range(1,len(self.classes)): # do not store bg
                self.idx_match_object_candneg[obj_cat] = np.where(cand_cat==obj_cat)[0]

            self.idx_match_subject_candneg = {}
            cand_cat = self.cand_negatives[:,2]
            for sub_cat in range(1,len(self.classes)): # do not store bg
                self.idx_match_subject_candneg[sub_cat] = np.where(cand_cat==sub_cat)[0]


    '''
    Methods to load image instance
    '''

    def image_filename(self, im_id):
        return self.db[im_id]['filename']


    def load_image(self, im_id):
        
        filename = self.image_filename(im_id)
        im = cv2.imread(osp.join(self.image_dir, filename),1)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB) # Convert BGR to RGB
        return im



    def get_obj_id_iccv(self, im_id, idx=None):
        """
        Return : (N,) annotation id of objects
        """
        obj_id = self.db[im_id]['obj_id_iccv']
        if idx is not None:
            if isinstance(idx, numbers.Number):
                idx = np.array([idx])
            obj_id = obj_id[idx]
        return obj_id


    def get_rel_id_iccv(self, im_id, idx=None):
        """
        Return : (N,) annotation id of objects
        """
        rel_id = self.db[im_id]['rel_id_iccv']
        if idx is not None:
            if isinstance(idx, numbers.Number):
                idx = np.array([idx])
            rel_id = rel_id[idx]
        return rel_id


    def get_labels_visualphrases(self, im_id, idx=None):
        """
        Return : (N,num_visualphrase)
        """
        labels_predicates = self.get_labels_predicates(im_id, idx=idx)
        pair_ids = self.get_pair_ids(im_id, idx=idx)
        obj_cat = self.get_gt_classes(im_id, idx=pair_ids[:,1])
        sub_cat = self.get_gt_classes(im_id, idx=pair_ids[:,0])

        # Return visual phrases labels
        labels_visualphrases = np.zeros((pair_ids.shape[0],self.num_visualphrases))
        for j in range(pair_ids.shape[0]):
            ind_rels = np.where(labels_predicates[j,:]==1)[0]
            for r in ind_rels:
                predicate = self.predicates.idx2word[r]
                subjname  = self.classes.idx2word[sub_cat[j]]
                objname   = self.classes.idx2word[obj_cat[j]]
                relation  = '-'.join([subjname,predicate, objname])
                if relation in self.visualphrases.words():
                    vp_cat = self.visualphrases(relation)
                    labels_visualphrases[j,vp_cat] = 1

        return labels_visualphrases


    def load_objectscores(self, im_id, cand_id=None):
        pair_ids = self.get_pair_ids(im_id, cand_id)
        objectscores = np.zeros((pair_ids.shape[0], 2, self.num_classes))

        # If the GT pair -> just put all scores to 1
        if self.db[im_id]['is_gt_pair'][cand_id]:
            objectscores.fill(1.0)

        else:
            subject_idx = self.get_obj_id_iccv(im_id, idx=pair_ids[:,0])
            object_idx = self.get_obj_id_iccv(im_id, idx=pair_ids[:,1])
            objectscores = np.zeros((pair_ids.shape[0], 2, self.num_classes))

            for j in range(pair_ids.shape[0]):
                idx = np.where(self.objectscores[:,0]==subject_idx[j])[0][0]
                objectscores[j,0,:] = self.objectscores[idx,1:]
                idx = np.where(self.objectscores[:,0]==object_idx[j])[0][0]
                objectscores[j,1,:] = self.objectscores[idx,1:] 

        return objectscores


    def load_appearance(self, im_id, cand_id=None):

        pair_ids         = self.get_pair_ids(im_id, cand_id)
        appearance_feats = np.zeros((pair_ids.shape[0],2,4096)) # no appearance features for union
        subject_idx      = self.get_obj_id_iccv(im_id, idx=pair_ids[:,0]) # attention: take iccv id !!
        object_idx       = self.get_obj_id_iccv(im_id, idx=pair_ids[:,1])

        # Is GT pair or candidate
        is_gt = self.db[im_id]['is_gt_pair'][cand_id]
        if np.all(is_gt)==1:
            candidates_name = 'annotated'
        elif np.all(is_gt)==0:
            if self.split in ['train','val','trainval'] or self.split in self.train_split_zeroshot:
                candidates_name = 'candidates'
            else:
                candidates_name = self.cand_test 
        else:   
            pdb.set_trace()

        if self.split in ['train','val','trainval'] or self.split in self.train_split_zeroshot: 
            features_path = osp.join(self.data_dir, 'train', candidates_name, 'features', 'raw', str(im_id) + '.mat')
        else:
            features_path = osp.join(self.data_dir, 'test', candidates_name, 'features', 'raw', str(im_id) + '.mat')

        features = sio.loadmat(features_path)
        features = features['features']


        for j in range(pair_ids.shape[0]):
            sub_id = np.where(subject_idx[j]==features[:,0])[0][0]
            appearance_feats[j,0,:] = features[sub_id,1:]
            obj_id = np.where(object_idx[j]==features[:,0])[0][0]
            appearance_feats[j,1,:] = features[obj_id,1:]

        if self.l2norm_input:
            appearance_feats[:,0,:] = appearance_feats[:,0,:] / np.linalg.norm(appearance_feats[:,0,:],axis=1)[:,None]
            appearance_feats[:,1,:] = appearance_feats[:,1,:] / np.linalg.norm(appearance_feats[:,1,:],axis=1)[:,None]


        return appearance_feats


    """
    Build database: contain all pairs : GT + candidates
    """

    def _build_db(self):
        db = {}
        for j in range(len(self.image_ids)):
            im_id = self.image_ids[j]
            db[im_id] = {}
            self._prep_db_entry(db[im_id])

            # At least fill up image_filename, width, height. Might not be annotations.
            filename = self.image_filenames[im_id-1]
            db[im_id]['filename'] = filename
            im = cv2.imread(osp.join(self.image_dir, filename),1)

            height, width, _ = im.shape
            db[im_id]['width'] = width
            db[im_id]['height'] = height

            
        # Fill with GT objects and pairs
        self._add_gt_annotations(db)

        # Fill with candidate objects
        self._populate_candidates(db)

        # Label candidate pairs
        self._label_candidates(db)

        return db


    def _prep_db_entry(self, entry):
        entry['filename'] = None
        entry['width']    = None
        entry['height']   = None
        entry['boxes']          = np.empty((0, 4), dtype=np.float32)
        entry['obj_classes']    = np.empty((0), dtype=np.int32) # will store the detected classes (with object detector)
        entry['obj_gt_classes'] = np.empty((0), dtype=np.int32) # store the GT classes
        entry['obj_scores']     = np.empty((0), dtype=np.float32) 
        entry['is_gt']          = np.empty((0), dtype=np.bool)
        entry['obj_id']         = np.empty((0), dtype=np.int32) 
        entry['obj_id_iccv']    = np.empty((0), dtype=np.int32)
        entry['pair_ids']       = np.empty((0,2), dtype=np.int32)
        entry['labels_r']       = lil_matrix((0, self.num_predicates))
        entry['is_gt_pair']     = np.empty((0), dtype=np.bool)
        entry['cand_id']        = np.empty((0), dtype=np.int32) # To identify candidate relation (relative indexing in image)
        entry['labels_sr']      = lil_matrix((0, len(self.subjectpredicates)))  
        entry['labels_ro']      = lil_matrix((0, len(self.objectpredicates))) 
        entry['pair_iou']       = np.empty((0,2), dtype=np.float32)
        entry['rel_id_iccv']    = np.empty((0), dtype=np.int32)

    """
    Pre-load VRD dataset
    """


    def _populate_candidates(self, db):
        """
        Load objects.mat either candidates or gt and convert in dict
        objects_py[im_id] = np.array([xmin, ymin, xmax, ymax, obj_cat, obj_id])
        """

        if self.split in ['train','val','trainval'] or self.split in self.train_split_zeroshot:
            objects_mat = sio.loadmat(osp.join(self.data_dir, 'train', 'candidates', 'objects.mat'), \
                                      squeeze_me=True, struct_as_record=False)
        else:
            objects_mat = sio.loadmat(osp.join(self.data_dir, 'test', self.cand_test, 'objects.mat'), \
                                      squeeze_me=True, struct_as_record=False)

        objects_mat = objects_mat['objects']

        for index in range(len(self.image_ids)):

            im_id = self.image_ids[index]
            idx = np.where(objects_mat.im_id==im_id)[0]

            if len(idx)>0:

                # Add the objects
                width, height = db[im_id]['width'], db[im_id]['height']
                
                x1 = objects_mat.object_box[idx][:,0] -1
                y1 = objects_mat.object_box[idx][:,1] -1
                x2 = objects_mat.object_box[idx][:,2] -1
                y2 = objects_mat.object_box[idx][:,3] -1
                
                x1, y1, x2, y2 = self.clip_xyxy_to_image(x1, y1, x2, y2, height, width)
                boxes = np.vstack((x1,y1,x2,y2)).T

                for j in range(len(idx)):
                    box = boxes[j,:]
                    obj_cat = objects_mat.obj_cat[idx[j]]
                    obj_id_annotation = objects_mat.obj_id[idx[j]]
                    if self.cand_test in ['annotated', 'gt-candidates']:
                        score = 1.0
                    else:
                        score = objects_mat.object_score[idx[j]]
                    # Transform x,y,w,h -> x,y,x2,y2
                    x1,y1,x2,y2 = box[0], box[1], box[2], box[3]

                    # Want boxes to have area at least 1
                    w = x2-x1+1
                    h = y2-y1+1
                    assert w>=1 and h>=1, 'Found candidates of area <1'

                    obj_id = np.max(db[im_id]['obj_id']) + 1 if len(db[im_id]['obj_id'])>0 else 0

                    # Get GT labels for subject/object boxes 
                    obj_gt_class  = 0
                    is_gt_objects = (db[im_id]['is_gt']==1)
                    gt_boxes      = db[im_id]['boxes'][is_gt_objects,:]
                    gt_classes    = db[im_id]['obj_classes'][is_gt_objects]

                    # Pre-init labels_sr, labels_ro to background (if no intersection with GT)
                    gt_labels_sr = db[im_id]['labels_sr'][is_gt_objects,:]
                    gt_labels_ro = db[im_id]['labels_ro'][is_gt_objects,:]

                    objname = 'background'
                    obj_labels_sr = np.zeros((1,len(self.subjectpredicates)))
                    relation = '-'.join([objname, 'no interaction'])
                    if relation in self.subjectpredicates.words():
                        ind_sr = self.subjectpredicates(relation)
                        obj_labels_sr[0,ind_sr] = 1

                    obj_labels_ro = np.zeros((1,len(self.objectpredicates)))
                    relation = '-'.join(['no interaction', objname])
                    if relation in self.objectpredicates.words():
                        ind_ro = self.objectpredicates(relation)
                        obj_labels_ro[0,ind_ro] = 1


                    if len(gt_boxes)>0:
                        ovl_gt = get_overlap(gt_boxes, np.array([x1,y1,x2,y2]))
                        id_max_ovl = np.argmax(ovl_gt)

                        # Label the box as positive for the GT with max overlap, providing that this overlap is above 0.5 
                        if ovl_gt[id_max_ovl]>0.5:
                            obj_gt_class = gt_classes[id_max_ovl]
                            obj_labels_sr = gt_labels_sr[id_max_ovl,:].toarray()
                            obj_labels_ro = gt_labels_ro[id_max_ovl,:].toarray()

                    # Append in database
                    db[im_id]['boxes']          = np.vstack((db[im_id]['boxes'], np.array(list([x1,y1,x2,y2])) ))
                    db[im_id]['obj_classes']    = np.hstack((db[im_id]['obj_classes'], np.array([obj_cat])))
                    db[im_id]['obj_gt_classes'] = np.hstack((db[im_id]['obj_gt_classes'], np.array([obj_gt_class])))
                    db[im_id]['obj_scores']     = np.hstack((db[im_id]['obj_scores'], np.array([score])))
                    db[im_id]['is_gt']          = np.hstack((db[im_id]['is_gt'], np.zeros((1), dtype=np.bool)))
                    db[im_id]['obj_id']         = np.hstack((db[im_id]['obj_id'], np.array([obj_id], dtype=np.int32)))
                    db[im_id]['obj_id_iccv']    = np.hstack((db[im_id]['obj_id_iccv'], np.array([obj_id_annotation], dtype=np.int32)))
                    db[im_id]['labels_sr']      = lil_matrix(np.vstack((db[im_id]['labels_sr'].toarray(), obj_labels_sr)))
                    db[im_id]['labels_ro']      = lil_matrix(np.vstack((db[im_id]['labels_ro'].toarray(), obj_labels_ro)))


    def _label_candidates(self, db):

        # Use same candidate pairs as in iccv17 to be exactly comparable
        if self.split in ['train','val','trainval'] or self.split in self.train_split_zeroshot:
            pairs_mat = sio.loadmat(osp.join(self.data_dir, 'train', 'candidates', 'pairs.mat'), \
                                    squeeze_me=True, struct_as_record=False)
        else:
            pairs_mat = sio.loadmat(osp.join(self.data_dir, 'test', self.cand_test, 'pairs.mat'), \
                                    squeeze_me=True, struct_as_record=False)

        pairs_mat = pairs_mat['pairs']

        for index in range(len(self.image_ids)):
            im_id = self.image_ids[index]
            idx = np.where(pairs_mat.im_id==im_id)[0]
            
            if len(idx)>0:
                boxes =db[im_id]['boxes']
                obj_classes = db[im_id]['obj_classes']
                idx_cand = np.where(db[im_id]['is_gt']==0)[0]
                idx_gt = np.where(db[im_id]['is_gt']==1)[0]
                id_annot_to_db = {db[im_id]['obj_id_iccv'][idx_cand[j]]:db[im_id]['obj_id'][idx_cand[j]] for j in range(len(idx_cand))}

                if len(idx_cand)==0:
                    continue

                if self.split in ['train', 'val', 'trainval'] or self.split in self.train_split_zeroshot:
                    if len(idx_gt)==0:
                        continue

                # Get the groundtruth annotations for this image
                is_gt_pair      = db[im_id]['is_gt_pair']
                gt_pair_ids     = db[im_id]['pair_ids']
                gt_pair_labels  = db[im_id]['labels_r'].toarray()
                gt_cand_id      = db[im_id]['cand_id']
                pair_iou        = db[im_id]['pair_iou']
                current_cand_id = np.max(gt_cand_id)+1 if len(gt_cand_id)>0 else 0

                # Form candidate pairs
                cand_pair_ids = np.zeros((len(idx),2), dtype=np.int32)
                for j in range(len(idx)):
                    cand_pair_ids[j,0] = id_annot_to_db[pairs_mat.sub_id[idx[j]]] 
                    cand_pair_ids[j,1] = id_annot_to_db[pairs_mat.obj_id[idx[j]]]        

                # Get rel_id in iccv17
                rel_id = pairs_mat.rel_id[idx]

                # Label subject-object relation
                idx_pos_pair = np.where(np.sum(gt_pair_labels[:,1:],1)>=1)[0]
                gt_pos_pair_ids = gt_pair_ids[idx_pos_pair,:]
                gt_pos_pair_labels = gt_pair_labels[idx_pos_pair,:]
                cand_pair_labels, cand_pair_iou = self.build_label(cand_pair_ids, gt_pos_pair_ids, gt_pos_pair_labels, boxes, obj_classes, 0.5)

                # Merge candidates with GT
                db[im_id]['pair_ids']    = np.vstack((gt_pair_ids, cand_pair_ids))
                db[im_id]['labels_r']    = lil_matrix(np.vstack((gt_pair_labels, cand_pair_labels)))
                db[im_id]['is_gt_pair']  = np.hstack((is_gt_pair, np.zeros((cand_pair_ids.shape[0]),dtype=np.bool)))
                db[im_id]['cand_id']     = np.hstack((gt_cand_id, current_cand_id+np.arange(cand_pair_ids.shape[0], dtype=np.int32) ))
                db[im_id]['rel_id_iccv'] = np.hstack((db[im_id]['rel_id_iccv'], np.array(rel_id).astype(int)))



    def build_label(self, cand_pair_ids, gt_pair_ids, gt_pair_labels, boxes, obj_classes, iou_pos):

        cand_pair_labels = np.zeros((len(cand_pair_ids), self.num_predicates))
        cand_pair_iou = np.zeros((len(cand_pair_ids),2))

        ids_subject = cand_pair_ids[:,0]
        ids_object = cand_pair_ids[:,1]

        # Scan the groundtruth relationships for this image and mark as positives candidates overlapping
        for j in range(gt_pair_ids.shape[0]):
            gt_sub      = gt_pair_ids[j,0]
            gt_obj      = gt_pair_ids[j,1]
            obj_cat     = obj_classes[gt_obj]
            subject_box = boxes[gt_sub,:]
            object_box  = boxes[gt_obj,:]

            # Filter candidates by category
            idx = np.where(obj_classes[ids_object]==obj_cat)[0]
            if len(idx)==0:
                continue

            # Overlap with candidates
            ovl_subject = get_overlap(boxes[ids_subject,:], subject_box)
            ovl_object  = get_overlap(boxes[ids_object[idx],:], object_box)

            # Fill overlap for both positives and negatives
            cand_pair_iou[:,0]   = np.maximum(cand_pair_iou[:,0], ovl_subject)
            cand_pair_iou[idx,1] = np.maximum(cand_pair_iou[idx,1], ovl_object)

            # Label as positives the candidates whose IoU > 0.5
            sub_ids_pos = np.where(ovl_subject>=iou_pos)[0]
            obj_ids_pos = np.where(ovl_object>=iou_pos)[0]

            # Label as positives if categories match, and IoU>0.5 for both subject and object 
            if len(sub_ids_pos)>0 and len(obj_ids_pos)>0:
                sub_ids_pos = ids_subject[sub_ids_pos]
                obj_ids_pos = ids_object[idx[obj_ids_pos]]

                for sub_id in sub_ids_pos:
                    for obj_id in obj_ids_pos:
                        cand_id = np.where(np.logical_and(ids_subject==sub_id, ids_object==obj_id))[0]
                        cand_pair_labels[cand_id,:] = np.maximum(cand_pair_labels[cand_id,:], gt_pair_labels[j,:]) # take max to have multilabeling

        # All candidates without intersection with a positive get assigned to background class
        id_bg = np.where(np.sum(cand_pair_labels,1)==0)[0]
        if len(id_bg)>0:
            cand_pair_labels[id_bg,0] = 1

        return cand_pair_labels, cand_pair_iou


    def _add_gt_annotations(self, db):
        """
        Load objects.mat either candidates or gt and convert in dict
        objects_py[im_id] = np.array([xmin, ymin, xmax, ymax, obj_cat, obj_id])
        """

        if self.split in ['train','val','trainval'] or self.split in self.train_split_zeroshot:
            objects_mat = sio.loadmat(osp.join(self.data_dir, 'train', 'annotated', 'objects.mat'), \
                                      squeeze_me=True, struct_as_record=False)
        else:
            objects_mat = sio.loadmat(osp.join(self.data_dir, 'test', 'annotated', 'objects.mat'), \
                                      squeeze_me=True, struct_as_record=False)

        objects_mat = objects_mat['objects']

        if self.split in ['train','val','trainval'] or self.split in self.train_split_zeroshot:
            pairs_mat = sio.loadmat(osp.join(self.data_dir, 'train', 'annotated', 'pairs.mat'), \
                                    squeeze_me=True, struct_as_record=False)
        else:
            pairs_mat = sio.loadmat(osp.join(self.data_dir, 'test', 'annotated', 'pairs.mat'), \
                                    squeeze_me=True, struct_as_record=False)

        pairs_mat = pairs_mat['pairs']


        for index in range(len(self.image_ids)):

            im_id = self.image_ids[index]
            idx = np.where(objects_mat.im_id==im_id)[0]

            if len(idx)>0:

                # Add the objects
                width, height = db[im_id]['width'], db[im_id]['height']
                
                x1 = objects_mat.object_box[idx][:,0] -1
                y1 = objects_mat.object_box[idx][:,1] -1
                x2 = objects_mat.object_box[idx][:,2] -1
                y2 = objects_mat.object_box[idx][:,3] -1
                 
                x1, y1, x2, y2 = self.clip_xyxy_to_image(x1, y1, x2, y2, height, width)
                boxes = np.vstack((x1,y1,x2,y2)).T

                obj_id_annotation = objects_mat.obj_id[idx]

                # Fill db
                db[im_id]['boxes']          = boxes
                db[im_id]['obj_classes']    = objects_mat.obj_cat[idx].astype(int)
                db[im_id]['obj_gt_classes'] = np.array(objects_mat.obj_cat[idx]).astype(int)
                db[im_id]['obj_scores']     = np.ones(len(idx))
                db[im_id]['is_gt']          = np.ones(len(idx), dtype=np.bool)
                db[im_id]['obj_id']         = np.arange(len(idx), dtype=np.int32)
                db[im_id]['obj_id_iccv']    = np.array(obj_id_annotation).astype(int)

                # Keep track of old obj_id
                id_annot_to_db = {db[im_id]['obj_id_iccv'][j]:db[im_id]['obj_id'][j] for j in range(len(obj_id_annotation))}

                # Add the relationships
                idx = np.where(pairs_mat.im_id==im_id)[0]
                sub_id = pairs_mat.sub_id[idx]
                obj_id = pairs_mat.obj_id[idx]
                rel_id = pairs_mat.rel_id[idx]
                sub_cat = pairs_mat.sub_cat[idx]
                obj_cat = pairs_mat.obj_cat[idx]
                rel_cat = pairs_mat.rel_cat[idx]
                all_relationships = np.empty((len(idx),4)) #[sub_id, obj_id, rel_cat]

                for p in range(len(idx)):
                    all_relationships[p,0] = id_annot_to_db[sub_id[p]]
                    all_relationships[p,1] = id_annot_to_db[obj_id[p]]
                    all_relationships[p,2] = rel_cat[p]  

                    # Add to vocab of visualphrases
                    subjname = self.classes.idx2word[sub_cat[p]]
                    objname = self.classes.idx2word[obj_cat[p]]
                    relname = self.predicates.idx2word[rel_cat[p]]
                    triplet = '-'.join([subjname, relname, objname])

                # Consider unique relationships
                relationships_unique = multilabel_transform(all_relationships, self.num_predicates) # Remove duplicates + binarize
                db[im_id]['pair_ids']    = relationships_unique[:,:2].astype(np.int32)
                db[im_id]['labels_r']    = lil_matrix(relationships_unique[:,2:])
                db[im_id]['is_gt_pair']  = np.ones((relationships_unique.shape[0]), dtype=np.bool)
                db[im_id]['cand_id']     = np.arange(relationships_unique.shape[0], dtype=np.int32)
                db[im_id]['pair_iou']    = np.ones((relationships_unique.shape[0],2), dtype=np.float32) # Iou of positive is 1 !
                db[im_id]['rel_id_iccv'] = -np.ones(relationships_unique.shape[0]).astype(int) 


        for index in range(len(self.image_ids)):

            im_id = self.image_ids[index]

            db[im_id]['labels_sr'] = lil_matrix((0, len(self.subjectpredicates))) 
            db[im_id]['labels_ro'] = lil_matrix((0, len(self.objectpredicates)))
            objects_ids = db[im_id]['obj_id']

            for o in range(len(objects_ids)):
                obj_id = objects_ids[o]
                obj_cat = db[im_id]['obj_classes'][obj_id]
                objname = self.classes.idx2word[obj_cat]

                # Find pairs where the object is involved as a subject
                idx = np.where(db[im_id]['pair_ids'][:,0]==obj_id)[0]

                labels_sr = np.zeros((1,len(self.subjectpredicates)))
                if len(idx)>0:
                    labels_predicates = db[im_id]['labels_r'][idx,:].toarray()
                    labels_predicates = np.max(labels_predicates,0) # the subject can interact with multiple subjects: get them all
                    ind_rels = np.where(labels_predicates[1:]==1)[0] # do not consider no_interaction class
                    out_of_vocab = 0
                    if len(ind_rels)>0:
                        for r in ind_rels:
                            predicate = self.predicates.idx2word[r+1]
                            relation = '-'.join([objname, predicate])
                            if relation in self.subjectpredicates.words():
                                ind_sr = self.subjectpredicates(relation)
                                labels_sr[0, ind_sr] = 1
                            else:
                                out_of_vocab = 1

                    # If no label AND not out of vocab (e.g. test split, unseen triplets), label as no_interaction
                    if np.sum(labels_sr)==0 and not out_of_vocab:
                        relation = '-'.join([objname, 'no interaction'])
                        ind_sr = self.subjectpredicates(relation)
                        labels_sr[0, ind_sr] = 1


                db[im_id]['labels_sr'] = lil_matrix(np.vstack((db[im_id]['labels_sr'].toarray(), labels_sr)))

                # Find pairs where the object is involved as an object
                idx = np.where(db[im_id]['pair_ids'][:,1]==obj_id)[0]

                labels_ro = np.zeros((1,len(self.objectpredicates)))
                if len(idx)>0:
                    labels_predicates = db[im_id]['labels_r'][idx,:].toarray()
                    labels_predicates = np.max(labels_predicates,0) # the subject can interact with multiple subjects: get them all
                    ind_rels = np.where(labels_predicates[1:]==1)[0]
                    out_of_vocab = 0
                    if len(ind_rels)>0:
                        for r in ind_rels:
                            predicate = self.predicates.idx2word[r+1]
                            relation = '-'.join([predicate, objname])
                            if relation in self.objectpredicates.words():
                                ind_ro = self.objectpredicates(relation)
                                labels_ro[0, ind_ro] = 1
                            else:
                                out_of_vocab = 1

                    if np.sum(labels_ro)==0 and not out_of_vocab:
                        # Label as no interaction
                        relation = '-'.join(['no interaction', objname])
                        ind_ro = self.objectpredicates(relation)
                        labels_ro[0, ind_ro] = 1

                db[im_id]['labels_ro'] = lil_matrix(np.vstack((db[im_id]['labels_ro'].toarray(), labels_ro)))
 


    def get_zeroshot_triplets(self):
        """
        Return list of zeroshot triplets
        """
        zeroshot_triplets = []
        filename = osp.join(self.data_dir, 'test', 'annotated', 'zeroshot_triplets.mat')
        if osp.exists(filename):
            tripletsname = sio.loadmat(filename, squeeze_me=True)
            tripletsname = tripletsname['triplets']

        for _,subjectname in enumerate(self.classes.words()):
            for _,predicate in enumerate(self.predicates.words()):
                for _,objectname in enumerate(self.classes.words()):

                    triplet = ' '.join([subjectname, predicate, objectname])
                    if triplet in tripletsname:

                        zeroshot_triplets.append('-'.join([subjectname, predicate, objectname]))

        return zeroshot_triplets


    def ids_zeroshot(self):
        """
        Index of zeroshot pairs in annotated
        """
        ids_zshot = sio.loadmat(osp.join(self.data_dir, self.split, 'annotated', 'ind_zeroshot.mat'))
        ids_zshot = ids_zshot['ind_zeroshot'][:,0]-1  # -1 matlab to python
        return ids_zshot


    def get_vocab_objects(self):
        objects = sio.loadmat(osp.join(self.data_dir, 'vocab_objects.mat'), squeeze_me=True)
        objects = objects['vocab_objects']
        vocab_objects = Vocabulary()
        vocab_objects.add_word('background', 'noun')
        for k in range(len(objects)):
            vocab_objects.add_word(objects[k], 'noun')

        return vocab_objects


    def get_vocab_predicates(self):

        predicates = sio.loadmat(osp.join(self.data_dir, 'vocab_predicates.mat'), squeeze_me=True)
        predicates = predicates['vocab_predicates']
        vocab_predicates = Vocabulary()
        vocab_predicates.add_word('no interaction', 'verb')
        for k in range(len(predicates)):
            vocab_predicates.add_word(predicates[k], 'verb')

        return vocab_predicates


    def get_vocab_subjectpredicates(self, visualphrases):

        subjectpredicates = Vocabulary()
        for visualphrase in visualphrases.words():
            triplet = visualphrase.split('-')
            subjectpredicate = '-'.join([triplet[0],triplet[1]])
            if subjectpredicate not in subjectpredicates.words():
                subjectpredicates.add_word(subjectpredicate, 'noun-verb')

        return subjectpredicates



    def get_vocab_objectpredicates(self, visualphrases):

        objectpredicates = Vocabulary()
        for visualphrase in visualphrases.words():
            triplet = visualphrase.split('-')
            objectpredicate = '-'.join([triplet[1],triplet[2]])
            if objectpredicate not in objectpredicates.words():
                objectpredicates.add_word(objectpredicate, 'verb-noun')

        return objectpredicates


    def get_vocab_all_triplets(self):
        triplets = Vocabulary()
        for subjectname in self.classes.words():
            for predicate in self.predicates.words():
                for objectname in self.classes.words():
                    triplet = '-'.join([subjectname, predicate, objectname])
                    triplets.add_word(triplet, 'noun-verb-noun')
        return triplets


    def create_val_split(self, proportion):
        '''
        Create validation split from training
        proportion in [0,1] is the proportion of training samples to dedicate to validation
        '''
        np.random.seed(0)

        np.random.shuffle(self.image_ids) 
        ind_cut = int(proportion*len(self.image_ids))
        ids_val = self.image_ids[0:ind_cut]
        ids_train = self.image_ids[ind_cut:]


        return ids_val, ids_train


    @staticmethod
    def clip_xyxy_to_image(x1, y1, x2, y2, height, width):
        x1 = np.minimum(width - 1., np.maximum(0., x1))
        y1 = np.minimum(height - 1., np.maximum(0., y1))
        x2 = np.minimum(width - 1., np.maximum(0., x2))
        y2 = np.minimum(height - 1., np.maximum(0., y2))
        return x1, y1, x2, y2



    def get_gt(self, subset='all'):
        """
        Generate gt_file with different options (depending on what we want to test)
        All the groundtruth positive relations we want to retrieve
        """

        gt = {}
        count = 0
        for im_id in self.image_ids:

            # Get all groundtruth relationships/objects in image
            cand_gt = self.filter_pairs_gt(im_id)

            if len(cand_gt)>0:

                # Init
                gt[im_id] = {}
                gt[im_id]['gt_label'] = np.empty((0,3))
                gt[im_id]['gt_box'] = np.empty((0,2,4))
                gt[im_id]['obj_id'] = np.empty((0,2))

                pair_ids    = self.get_pair_ids(im_id, cand_gt)
                subject_box = self.get_boxes(im_id, pair_ids[:,0])
                object_box  = self.get_boxes(im_id, pair_ids[:,1])
                sub_cat     = self.get_classes(im_id, pair_ids[:,0])
                obj_cat     = self.get_classes(im_id, pair_ids[:,1])
                sub_id_iccv = self.get_obj_id_iccv(im_id, pair_ids[:,0])
                obj_id_iccv = self.get_obj_id_iccv(im_id, pair_ids[:,1])
                labels_r    = self.get_labels_predicates(im_id, cand_gt)

                for l in range(pair_ids.shape[0]): 
                    idx = np.where(labels_r[l,:]==1)[0]
                    for j in range(len(idx)):
                        rel_cat = idx[j]
                        # If not no_interaction
                        if rel_cat!=0:

                            # Filter subset: zeroshot / predicate type etc
                            is_in_subset = self.filter_subset(sub_cat[l], rel_cat, obj_cat[l], subset)
                            if is_in_subset:
                                gt[im_id]['gt_label'] = np.vstack((gt[im_id]['gt_label'], np.array([sub_cat[l], rel_cat, obj_cat[l]])))
                                gt[im_id]['gt_box'] = np.vstack((gt[im_id]['gt_box'], np.stack([subject_box[l,:][None,:], object_box[l,:][None,:]],1)))
                                gt[im_id]['obj_id'] = np.vstack((gt[im_id]['obj_id'], np.array([sub_id_iccv[l], obj_id_iccv[l]])))
                                count += 1

        print('Found %d groundtruth relationships to evaluate on' %count)


        return gt


    def computeArea(self, bb):
        return max(0, bb[2] - bb[0] + 1) * max(0, bb[3] - bb[1] + 1)

    def computeIoU(self, bb1, bb2):
        ibb = [max(bb1[0], bb2[0]), \
            max(bb1[1], bb2[1]), \
            min(bb1[2], bb2[2]), \
            min(bb1[3], bb2[3])]
        iArea = self.computeArea(ibb)
        uArea = self.computeArea(bb1) + self.computeArea(bb2) - iArea
        return (iArea + 0.0) / uArea


    def computeOverlap(self, detBBs, gtBBs):
        aIoU = self.computeIoU(detBBs[0, :], gtBBs[0, :])
        bIoU = self.computeIoU(detBBs[1, :], gtBBs[1, :])
        return min(aIoU, bIoU)      


    def get_occurrences(self, split):
        """
        Scan the cand_positives to get the occurrences -> number of positive candidates <> number of positives annotated (because of duplicate boxes)
        """
        cand_positives = pickle.load(open(osp.join(self.data_dir, 'cand_positives_' + split + '.pkl'),'rb'))

        occurrences = {}
        for j in range(cand_positives.shape[0]):

            im_id = cand_positives[j,0]
            cand_id = cand_positives[j,1]
            sub_cat = np.where(self.get_labels_subjects(im_id, cand_id))[1][0] #only 1 subject category (?)
            obj_cat = np.where(self.get_labels_objects(im_id, cand_id))[1][0] # only 1 object category
            rel_cats = np.where(self.get_labels_predicates(im_id, cand_id))[1]

            for _,rel_cat in enumerate(rel_cats):
                subjectname = self.vocab_grams['s'].idx2word[sub_cat]
                objectname = self.vocab_grams['o'].idx2word[obj_cat]
                predicate = self.vocab_grams['r'].idx2word[rel_cat]
                tripletname = '-'.join([subjectname, predicate, objectname])

                if tripletname in occurrences.keys():
                    occurrences[tripletname] += 1
                else:
                    occurrences[tripletname] = 1

        return occurrences



    def get_occurrences_precomp(self, split):
        """ 
        Get number of triplets annotated in split: only work on trainval.
        TODO: make file for all split 
        """
        triplets_remove = []
        if split in self.train_split_zeroshot:
            split, zeroshotset = split.split('_')
            triplets_remove = pickle.load(open(osp.join(self.data_dir,'triplet_queries.pkl'), 'rb'))

        filename = osp.join(self.data_dir, 'statistics', 'occurrences.csv')
        count = 0
        occurrences = {}
        with open(filename) as f:
            reader = csv.DictReader(f)
            for line in reader:
                occ_split = line['occ_' + split]
                action_name = line['action_name']
                occurrences[action_name] = int(occ_split)
                count += 1

        for triplet_remove in triplets_remove:
            occurrences[triplet_remove] = 0


        return occurrences
   


    def eval(self, detections, triplets, min_overlap=0.5):
        """
        detections: 
        For each image im_id (in order), each triplet r:
        detections[r][im_id] : [subject_box, object_box, score, cand_id]
        triplets: 
        List of triplets we want to compute AP on
        Inspired by HICO-DET evaluation code + jwyang-faster-rcnn eval code Visual Genome
        """
        ap = np.zeros((len(triplets),))
        recall = np.zeros((len(triplets),))

        gt = {}
        npos = {} # number of positives for each triplet
        for triplet in triplets:
            gt[triplet] = {im_id:np.empty((0,8)) for im_id in self.image_ids}
            npos[triplet] = 0

        """ Get the GT annotations for each triplet """
        gt_file = osp.join(self.data_dir, 'gt_file_iccv17_test.pkl')
        pairs_mat = sio.loadmat(osp.join(self.data_dir, 'test', 'annotated', 'pairs.mat'), squeeze_me=True, struct_as_record=False)
        pairs_mat = pairs_mat['pairs']

        for index in range(len(self.image_ids)):

            im_id = self.image_ids[index] 
            indices = np.where(pairs_mat.im_id==im_id)[0]

            for l in range(len(indices)):

                idx = indices[l]

                sub_cat     = pairs_mat.sub_cat[idx]
                subjectname = self.classes.idx2word[sub_cat]

                obj_cat    = pairs_mat.obj_cat[idx]
                objectname = self.classes.idx2word[obj_cat]

                rel_cat    = pairs_mat.rel_cat[idx]
                predicate  = self.predicates.idx2word[rel_cat]

                triplet    = '-'.join([subjectname, predicate, objectname])

                if triplet in triplets:

                    x1_s = pairs_mat.subject_box[idx,0] -1
                    y1_s = pairs_mat.subject_box[idx,1] -1
                    x2_s = pairs_mat.subject_box[idx,2] -1
                    y2_s = pairs_mat.subject_box[idx,3] -1
                    subject_box = np.array([x1_s,y1_s,x2_s,y2_s])

                    x1_o = pairs_mat.object_box[idx,0] -1
                    y1_o = pairs_mat.object_box[idx,1] -1
                    x2_o = pairs_mat.object_box[idx,2] -1
                    y2_o = pairs_mat.object_box[idx,3] -1
                    object_box = np.array([x1_o,y1_o,x2_o,y2_o])

                    gt[triplet][im_id] = np.vstack((gt[triplet][im_id], np.hstack((subject_box, object_box))))
                    npos[triplet] += 1


        """ Match the detections """

        for t in range(len(triplets)):
            triplet = triplets[t]

            #vp_cat = self.visualphrases.word2idx[triplet]
            vp_cat = t  # ATTENTION : might be a pb here
            gt_triplet = gt[triplet]

            # Get the detections for the triplet and sort by decreasing conf score across all images
            dets_triplet = np.empty((0,10)) #im_id, subject_box, object_box, score
            for i in range(len(self.image_ids)):
                im_id = self.image_ids[i]
                dets_im = np.hstack((im_id*np.ones((detections[vp_cat][i].shape[0],1)), detections[vp_cat][i][:,:9]))
                dets_triplet = np.vstack((dets_triplet, dets_im))
            idx_sort = np.argsort(dets_triplet[:,9])[::-1]
            dets_triplet = dets_triplet[idx_sort,:]

            tp = np.zeros((len(dets_triplet),1))
            fp = np.zeros((len(dets_triplet),1))
            detected = {im_id:np.zeros((gt_triplet[im_id].shape[0],1)) for im_id in gt_triplet.keys()}

            for d in range(dets_triplet.shape[0]):
                im_id = dets_triplet[d,0]
                bbox_s = dets_triplet[d,1:5]
                bbox_o = dets_triplet[d,5:9]

                ov_max = -np.inf
                gt_im = gt_triplet[im_id]

                for j in range(gt_im.shape[0]):

                    bbox_gt_s = gt_im[j,:4]
                    bbox_gt_o = gt_im[j,4:8]

                    # Compare subject
                    bi_s = np.array([   max(bbox_s[0],bbox_gt_s[0]),\
                                        max(bbox_s[1],bbox_gt_s[1]),\
                                        min(bbox_s[2],bbox_gt_s[2]),\
                                        min(bbox_s[3],bbox_gt_s[3])])

                    iw_s = bi_s[2]-bi_s[0]+1
                    ih_s = bi_s[3]-bi_s[1]+1

                    if iw_s > 0 and ih_s > 0:
                        # compute overlap as area of intersection / area of union
                        ua_s = (bbox_gt_s[3]-bbox_gt_s[1]+1)*(bbox_gt_s[2]-bbox_gt_s[0]+1) + \
                                (bbox_s[3]-bbox_s[1]+1)*(bbox_s[2]-bbox_s[0]+1) -\
                                iw_s*ih_s
                        ov_s = iw_s*ih_s/ua_s
                    else:
                        ov_s = 0


                    # Compare object
                    bi_o = np.array([   max(bbox_o[0],bbox_gt_o[0]),\
                                        max(bbox_o[1],bbox_gt_o[1]),\
                                        min(bbox_o[2],bbox_gt_o[2]),\
                                        min(bbox_o[3],bbox_gt_o[3])])

                    iw_o = bi_o[2]-bi_o[0]+1
                    ih_o = bi_o[3]-bi_o[1]+1

                    if iw_o > 0 and ih_o > 0:
                        # compute overlap as area of intersection / area of union
                        ua_o = (bbox_gt_o[3]-bbox_gt_o[1]+1)*(bbox_gt_o[2]-bbox_gt_o[0]+1) + \
                                (bbox_o[3]-bbox_o[1]+1)*(bbox_o[2]-bbox_o[0]+1) -\
                                iw_o*ih_o
                        ov_o = iw_o*ih_o/ua_o
                    else:
                        ov_o = 0


                    # Min overlap subject, object
                    min_ov = min(ov_s, ov_o)
                    if min_ov > ov_max:
                        ov_max = min_ov
                        j_max = j

                if ov_max >= min_overlap:
                    if detected[im_id][j_max]==0:
                        tp[d] = 1
                        detected[im_id][j_max] = 1
                    else:
                        fp[d] = 1
                else:
                    fp[d] = 1


            # compute precision recall
            fp = np.cumsum(fp)
            tp = np.cumsum(tp)
            rec = tp / float(npos[triplet])
            # ground truth
            prec = tp / (tp + fp)
            #prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
            ap[t] = self.voc_ap(rec, prec, use_07_metric=False)
            if len(rec)>0:
                recall[t] = max(rec)
            else:
                recall[t] = 0

        return ap, recall

    @staticmethod
    def voc_ap(rec, prec, use_07_metric=False):
      """ ap = voc_ap(rec, prec, [use_07_metric])
      Compute VOC AP given precision and recall.
      If use_07_metric is true, uses the
      VOC 07 11 point method (default:False).
      """
      if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
          if np.sum(rec >= t) == 0:
            p = 0
          else:
            p = np.max(prec[rec >= t])
          ap = ap + p / 11.
      else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
          mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

        return ap

