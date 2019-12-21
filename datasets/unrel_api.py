from __future__ import division
import os.path as osp
import scipy.io as sio
import numpy as np
import cv2
import scipy.misc
from utils import multilabel_transform, get_overlap, filter_small_boxes, Vocabulary
from scipy.sparse import lil_matrix
import scipy.sparse as sparse
import cPickle as pickle
import matplotlib
import copy
import time
import numbers
from Dataset import BaseDataset


class Unrel(BaseDataset):

    def __init__(self, data_dir, image_dir, split, cand_dir, cand_test='candidates', thresh_file=None, use_gt=False, add_gt=True, train_mode=True, jittering=False, filter_images=True, nms_thresh=0.5, store_ram=[], l2norm_input=False, neg_GT=False):
        super(Unrel, self).__init__()
        """ Only test split for UnRel """

        self.data_name    = 'unrel' 
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
        self.l2norm_input = l2norm_input

        self.d_appearance = 4096

        assert self.split=='test', 'Only test split for UnRel'

 
        self.image_ids = np.loadtxt(osp.join(data_dir, 'splits', self.split + '.ids'), dtype=int) 

        image_filenames = sio.loadmat(osp.join(self.data_dir, 'image_filenames_' + self.split + '.mat'), squeeze_me=True)    
        self.image_filenames = image_filenames['image_filenames']

        # Vocabulary
        self.classes        = self.get_vocab_objects()
        self.num_classes    = len(self.classes)
        self.predicates     = self.get_vocab_predicates()
        self.num_predicates = len(self.predicates)

        # Vocab of visualphrases 
        self.visualphrases     = pickle.load(open(osp.join(self.data_dir, 'visualphrases_trainval.pkl'),'rb'))
        self.num_visualphrases = len(self.visualphrases)
        self.subjectpredicates = self.get_vocab_subjectpredicates(self.visualphrases)
        self.objectpredicates  = self.get_vocab_objectpredicates(self.visualphrases)


        # Build or load database
        db_name = 'db_' + self.split + '_' + self.cand_test +'.pkl'
        if osp.exists(osp.join(self.data_dir, db_name)):
            self.db = pickle.load(open(osp.join(self.data_dir, db_name),'rb'))
        else:
            self.db = self._build_db()
            pickle.dump(self.db, open(osp.join(self.data_dir, db_name),'wb'))

        # Object scores
        self.objectscores = sio.loadmat(osp.join(self.data_dir, 'test', 'candidates', 'objectscores.mat'), squeeze_me=True)
        self.objectscores = self.objectscores['scores']


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
        self.idx_to_vocab = self.get_idx_in_vocab(self.vocab_grams, self.vocab_grams['all']) # get idx of vocab_grams in vocab_all (to access pre-computed word embeddings)


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


        self.candidates = self.get_test_candidates(use_gt=self.use_gt, thresh_file=self.thresh_file)


    """
    Methods to load image instance
    """

    def image_filename(self, im_id):
        return self.db[im_id]['filename']


    def load_image(self, im_id):
        
        filename = self.image_filename(im_id)
        im = cv2.imread(osp.join(self.image_dir, filename),1)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB) # Convert BGR to RGB
        return im

    def image_size(self, im_id):
        width = self.db[im_id]['width']
        height = self.db[im_id]['height']

        return width, height

    def get_obj_id(self, im_id, idx=None):
        """
        Return : (N,) annotation id of objects
        """
        obj_id = self.db[im_id]['obj_id']
        if idx is not None:
            if isinstance(idx, numbers.Number):
                idx = np.array([idx])
            obj_id = obj_id[idx]
        return obj_id


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
                objname = self.classes.idx2word[obj_cat[j]]
                subjname = self.classes.idx2word[sub_cat[j]]
                relation = '-'.join([subjname,predicate, objname])
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

        appearance_feats = self.load_appearance_iccv(im_id, cand_id)

        return appearance_feats


    def load_appearance_iccv(self, im_id, cand_id=None):

        pair_ids = self.get_pair_ids(im_id, cand_id)
        appearance_feats = np.zeros((pair_ids.shape[0],2,4096)) # no appearance features for union
        subject_idx = self.get_obj_id_iccv(im_id, idx=pair_ids[:,0])
        object_idx = self.get_obj_id_iccv(im_id, idx=pair_ids[:,1])

        # Is GT pair or candidate
        is_gt = self.db[im_id]['is_gt_pair'][cand_id]
        if np.all(is_gt)==1:
            candidates_name = 'annotated'
        elif np.all(is_gt)==0:
            candidates_name = self.cand_test
        else:
            pdb.set_trace()

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
        entry['obj_scores']     = np.empty((0), dtype=np.float32) # Later: for detections can take the scores over all classes
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

        if self.split in ['train','val','trainval']:
            objects_mat = sio.loadmat(osp.join(self.data_dir, 'train', 'candidates', 'objects.mat'), squeeze_me=True, struct_as_record=False)
        else:
            objects_mat = sio.loadmat(osp.join(self.data_dir, 'test', self.cand_test, 'objects.mat'), squeeze_me=True, struct_as_record=False)

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

                    # Get GT labels for subject/object boxes (could be used eventually to refine detections on this dataset)
                    obj_gt_class = 0
                    is_gt_objects = (db[im_id]['is_gt']==1)
                    gt_boxes = db[im_id]['boxes'][is_gt_objects,:]
                    gt_classes = db[im_id]['obj_classes'][is_gt_objects]

                    # Pre-init labels_sr, labels_ro to background (if no intersection with GT)

                    gt_labels_sr = db[im_id]['labels_sr'][is_gt_objects,:]
                    gt_labels_ro = db[im_id]['labels_ro'][is_gt_objects,:]

                    objname = '__background__'
                    obj_labels_sr = np.zeros((1,len(self.subjectpredicates)))
                    relation = '-'.join([objname, 'no_interaction'])
                    if relation in self.subjectpredicates.words():
                        ind_sr = self.subjectpredicates(relation)
                        obj_labels_sr[0,ind_sr] = 1

                    obj_labels_ro = np.zeros((1,len(self.objectpredicates)))
                    relation = '-'.join(['no_interaction', objname])
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
        if self.split in ['train','val','trainval']:
            pairs_mat = sio.loadmat(osp.join(self.data_dir, 'train', 'candidates', 'pairs.mat'), squeeze_me=True, struct_as_record=False)
        else:
            pairs_mat = sio.loadmat(osp.join(self.data_dir, 'test', self.cand_test, 'pairs.mat'), squeeze_me=True, struct_as_record=False)

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

                # Keep also images for which no GT (compatibility with iccv17)
                if len(idx_cand)==0:
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
                idx_pos_pair                    = np.where(np.sum(gt_pair_labels[:,1:],1)>=1)[0]
                gt_pos_pair_ids                 = gt_pair_ids[idx_pos_pair,:]
                gt_pos_pair_labels              = gt_pair_labels[idx_pos_pair,:]
                cand_pair_labels, cand_pair_iou = self.build_label(cand_pair_ids, gt_pos_pair_ids, gt_pos_pair_labels, boxes, obj_classes, 0.5)

                # Merge candidates with GT
                db[im_id]['pair_ids']    = np.vstack((gt_pair_ids, cand_pair_ids))
                db[im_id]['labels_r']    = lil_matrix(np.vstack((gt_pair_labels, cand_pair_labels)))
                db[im_id]['is_gt_pair']  = np.hstack((is_gt_pair, np.zeros((cand_pair_ids.shape[0]),dtype=np.bool)))
                db[im_id]['cand_id']     = np.hstack((gt_cand_id, current_cand_id+np.arange(cand_pair_ids.shape[0], dtype=np.int32) ))
                db[im_id]['rel_id_iccv'] = np.hstack((db[im_id]['rel_id_iccv'], np.array(rel_id).astype(int)))


    def build_label(self, cand_pair_ids, gt_pair_ids, gt_pair_labels, boxes, obj_classes, iou_pos):

        cand_pair_labels = np.zeros((len(cand_pair_ids), self.num_predicates))
        cand_pair_iou    = np.zeros((len(cand_pair_ids),2))

        ids_subject = cand_pair_ids[:,0]
        ids_object  = cand_pair_ids[:,1]

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

        if self.split in ['train','val','trainval']:
            objects_mat = sio.loadmat(osp.join(self.data_dir, 'train', 'annotated', 'objects.mat'), squeeze_me=True, struct_as_record=False)
        else:
            objects_mat = sio.loadmat(osp.join(self.data_dir, 'test', 'annotated', 'objects.mat'), squeeze_me=True, struct_as_record=False)

        objects_mat = objects_mat['objects']

        if self.split in ['train','val','trainval']:
            pairs_mat = sio.loadmat(osp.join(self.data_dir, 'train', 'annotated', 'pairs.mat'), squeeze_me=True, struct_as_record=False)
        else:
            pairs_mat = sio.loadmat(osp.join(self.data_dir, 'test', 'annotated', 'pairs.mat'), squeeze_me=True, struct_as_record=False)
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
                all_relationships = np.empty((len(idx),4)) #[sub_id, obj_id, rel_cat]
                idx     = np.where(pairs_mat.im_id==im_id)[0]
                sub_id  = pairs_mat.sub_id[idx]
                obj_id  = pairs_mat.obj_id[idx]
                rel_id  = pairs_mat.rel_id[idx]
                sub_cat = pairs_mat.sub_cat[idx]
                obj_cat = pairs_mat.obj_cat[idx]
                rel_cat = pairs_mat.rel_cat[idx]
                all_relationships = np.empty((len(idx),4)) #[sub_id, obj_id, rel_cat]

                for p in range(len(idx)):
                    all_relationships[p,0] = id_annot_to_db[sub_id[p]]
                    all_relationships[p,1] = id_annot_to_db[obj_id[p]]
                    all_relationships[p,2] = rel_cat[p] # attention: introduce no_interaction class 

                    # Add to vocab of visualphrases
                    subjname = self.classes.idx2word[sub_cat[p]]
                    objname  = self.classes.idx2word[obj_cat[p]]
                    relname  = self.predicates.idx2word[rel_cat[p]]
                    triplet  = '-'.join([subjname, relname, objname])

                # Consider unique relationships
                relationships_unique     = multilabel_transform(all_relationships, self.num_predicates) # Remove duplicates + binarize
                db[im_id]['pair_ids']    = relationships_unique[:,:2].astype(np.int32)
                db[im_id]['labels_r']    = lil_matrix(relationships_unique[:,2:])
                db[im_id]['is_gt_pair']  = np.ones((relationships_unique.shape[0]), dtype=np.bool)
                db[im_id]['cand_id']     = np.arange(relationships_unique.shape[0], dtype=np.int32)
                db[im_id]['pair_iou']    = np.ones((relationships_unique.shape[0],2), dtype=np.float32) # Iou of positive is 1 !
                db[im_id]['rel_id_iccv'] = -np.ones(relationships_unique.shape[0]).astype(int) # ATTENTION: no correspondency because we call multilabel_transform !!!!


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
                    if len(ind_rels)>0:
                        for r in ind_rels:
                            predicate = self.predicates.idx2word[r+1]
                            relation = '-'.join([objname, predicate])
                            ind_sr = self.subjectpredicates(relation)
                            labels_sr[0, ind_sr] = 1

                    # If no label, label as no_interaction
                    if np.sum(labels_sr)==0:
                        relation = '-'.join([objname, 'no_interaction'])
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
                    if len(ind_rels)>0:
                        for r in ind_rels:
                            predicate = self.predicates.idx2word[r+1]
                            relation = '-'.join([predicate, objname])
                            ind_ro = self.objectpredicates(relation)
                            labels_ro[0, ind_ro] = 1

                    if np.sum(labels_ro)==0:
                        # Label as no interaction
                        relation = '-'.join(['no_interaction', objname])
                        ind_ro = self.objectpredicates(relation)
                        labels_ro[0, ind_ro] = 1

                db[im_id]['labels_ro'] = lil_matrix(np.vstack((db[im_id]['labels_ro'].toarray(), labels_ro)))
 

    def get_zeroshot_triplets(self):
        """
        Return a list of zeroshot triplets (pre-computed)
        """
        zeroshot_triplets = []
        filename = osp.join(self.data_dir, self.split, 'annotated', 'zeroshot_triplets.mat')
        if osp.exists(filename):
            zeroshot_triplets = sio.loadmat(filename, squeeze_me=True)
            zeroshot_triplets = zeroshot_triplets['triplets']
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
        vocab_objects.add_word('__background__', 'noun')
        for k in range(len(objects)):
            vocab_objects.add_word(objects[k], 'noun')

        return vocab_objects


    def get_vocab_predicates(self):

        predicates = sio.loadmat(osp.join(self.data_dir, 'vocab_predicates.mat'), squeeze_me=True)
        predicates = predicates['vocab_predicates']
        vocab_predicates = Vocabulary()
        vocab_predicates.add_word('no_interaction', 'verb')
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
        """
        Create validation split from training
        proportion in [0,1] is the proportion of training samples to dedicate to validation
        """
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




