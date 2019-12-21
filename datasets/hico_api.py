import __init__
import os.path as osp
import os, json
import numpy as np
import cv2
import scipy.misc
from pycocotools.coco import COCO
from scipy.sparse import lil_matrix
import numbers
import csv
import cPickle as pickle
from datasets.utils import multilabel_transform, get_overlap, filter_small_boxes, Vocabulary
from datasets.Dataset import BaseDataset


class Hico(BaseDataset):

    def __init__(self, data_dir, image_dir, split, cand_dir, thresh_file=None, use_gt=False, add_gt=True, train_mode=True, jittering=False, nms_thresh=0.3, store_ram=[], l2norm_input=False, neg_GT=True): 
        super(Hico, self).__init__()

        self.data_name = 'hico'
        self.split = split
        self.data_dir = data_dir
        self.image_dir = image_dir
        self.cand_dir = cand_dir
        self.use_gt = use_gt
        self.add_gt = add_gt
        self.thresh_file = thresh_file
        self.jittering = jittering
        self.nms_thresh = nms_thresh
        self.store_ram = store_ram
        self.l2norm_input = l2norm_input

        self.d_appearance = 1024 # dimension of pre-extracted appearance feature (change according to your object detector)


        # Add options processing db
        self.neg_GT = True # whether to form negative pairs from GT at training or not
        self.iou_pos = 0.5 # iou threshold with GT above which a candidate is considered as positive
        self.iou_neg = 0.5 # iou threshold below which a candidate is considered as negative


        # Init COCO to get vocabulary of objects
        self.COCO = COCO(osp.join(self.data_dir, 'annotations_json', 'instances_train2014.json'))
        self._init_coco()

        # Load vocabulary of relations (triplets=visualphrases)
        self.actions           = json.load(open(osp.join(self.data_dir, 'annotations_json', 'actions.json'), 'rb'))
        self.visualphrases     = self.get_vocab_visualphrases(self.actions)
        self.num_visualphrases = len(self.visualphrases)


        # Define intermediate vocabulary: predicates, bigrams, trigrams
        self.predicates        = self.get_vocab_predicates(self.visualphrases)
        self.num_predicates    = len(self.predicates)
        self.subjectpredicates = self.get_vocab_subjectpredicates(self.visualphrases)
        self.objectpredicates  = self.get_vocab_objectpredicates(self.visualphrases)


        # Load image ids for split (txt file)
        self.train_split_zeroshot = ['trainval_zeroshottriplet','train_zeroshottriplet']

        # Load image ids
        self.image_ids = self.load_image_ids(split)


        # Load image filenames
        self.image_filenames = self.load_image_filenames(split)


        # Build database
        print('Building database from GT annotations...')

        if split in self.train_split_zeroshot:
            self.db = pickle.load(open(osp.join(self.data_dir, 'db_' + split.split('_')[0] + '.pkl'),'rb')) 
        else:

            if osp.exists(osp.join(self.data_dir, 'db_' + self.split + '.pkl')):
                self.db = pickle.load(open(osp.join(self.data_dir, 'db_' + self.split + '.pkl'),'rb')) 
            else:
                # Load the annotations
                if split in ['debug', 'train', 'val', 'trainval'] or split in self.train_split_zeroshot:
                    annotations = json.load(open(osp.join(self.data_dir, 'annotations_json', 'annotations_trainval.json'), 'rb'))
                elif split=='test':
                    annotations = json.load(open(osp.join(self.data_dir, 'annotations_json', 'annotations_test.json'), 'rb'))
                else:
                    print('Incorrect name split')
                    return
         
                # Build database
                self.db = self._build_db(annotations) 
                self.populate_candidates()           
                self.label_candidates()
                pickle.dump(self.db, open(osp.join(self.data_dir, 'db_' + self.split + '.pkl'),'wb'))

        # Some training images are flipped. We remove them.
        im_ids = []
        if self.split in ['train','trainval'] or self.split in self.train_split_zeroshot:
            im_ids = np.array([18656,31992,27273,19110,28274], dtype=int)
            self.image_ids = np.setdiff1d(self.image_ids, im_ids)


        # Filter detections (per-class threshold to maintain precision 0.3 measured on COCO dataset)
        if self.thresh_file:
            self.dets_thresh = np.load(osp.join(self.cand_dir, self.thresh_file + '.npy'))
        else:
            self.dets_thresh = None

        # Load candidates for training
        if train_mode:
            if osp.exists(osp.join(self.data_dir, 'cand_positives_' + split + '.pkl')):

                self.cand_positives = pickle.load(open(osp.join(self.data_dir, 'cand_positives_' + split + '.pkl'),'rb'))
                self.cand_negatives = pickle.load(open(osp.join(self.data_dir, 'cand_negatives_' + split + '.pkl'),'rb'))

            else:
                self.cand_positives, self.cand_negatives = self.get_training_candidates(use_gt=self.use_gt, add_gt=self.add_gt, thresh_file=self.thresh_file)
                pickle.dump(self.cand_positives, open(osp.join(self.data_dir, 'cand_positives_' + split + '.pkl'), 'wb'))
                pickle.dump(self.cand_negatives, open(osp.join(self.data_dir, 'cand_negatives_' + split + '.pkl'), 'wb'))

        else:
            self.candidates = self.get_test_candidates(use_gt=self.use_gt, thresh_file=self.thresh_file, nms_thresh=self.nms_thresh)



        # Vocab wrapper (use POS tag as can have homonyms verb/noun)
        self.vocab = self.build_vocab(self.classes, self.predicates)
        pickle.dump(self.vocab.idx2word.values(), open(osp.join(self.data_dir, 'vocab' + '.pkl'), 'wb')) 


        self.vocab_grams = {'s':self.classes,
                            'o':self.classes,
                            'r':self.predicates,
                            #'sr':self.subjectpredicates, # for expe bigram uncomment
                            #'ro':self.objectpredicates, # for expe bigram uncomment
                            'sr':[], # attention for expe coco-a uncomment
                            'ro':[], # attention for expe coco-a uncomment
                            'sro':self.visualphrases,
                            'all':self.vocab,
                            'vp_frequent':[]}

        self.idx_sro_to = self.get_idx_between_vocab(self.vocab_grams['sro'], self.vocab_grams)
        self.idx_to_vocab = self.get_idx_in_vocab(self.vocab_grams, self.vocab_grams['all']) # get idx of vocab_grams in vocab_all (to access pre-computed word embeddings)

        # Pre-trained word embeddings for subject/object/verb
        self.word_embeddings = pickle.load(open(osp.join(self.data_dir, 'pretrained_embeddings_w2v.pkl'), 'rb'))


        if self.l2norm_input:
            if (np.linalg.norm(self.word_embeddings,axis=1)==0).any():
                raise Exception('At least one word embedding vector is 0 (would cause nan after normalization)')
            self.word_embeddings = self.word_embeddings / np.linalg.norm(self.word_embeddings,axis=1)[:,None]


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
        Speed-up 2 : pre-compute the np.where(cand_cat==obj_cat) in dset.cand_negatives (speed-up sampler in BaseLoader)
        """
        if train_mode:
            cand_cat = self.cand_negatives[:,3]
            self.idx_match_object_candneg = {}
            for obj_cat in range(1,len(self.classes)): # do not store bg
                self.idx_match_object_candneg[obj_cat] = np.where(cand_cat==obj_cat)[0]



    """
    Methods to load instance
    """

    def load_image_ids(self, split):
        path = osp.join(self.data_dir, 'annotations_json', '%s.ids')
        if split=='debug':
            image_ids = np.loadtxt(open(path%'train','r'))
            image_ids = self.image_ids[0:10]
        elif split in self.train_split_zeroshot:
            image_ids = np.loadtxt(open(path%split.split('_')[0],'r'))
        else:
            image_ids = np.loadtxt(open(path%split,'r'))

        image_ids = image_ids.astype(np.int32)

        return image_ids


    def load_image_filenames(self, split):
        """ Load image filenames """
        path = osp.join(self.data_dir, 'annotations_json','image_filenames_%s.json')
        if split=='debug':
            image_filenames = json.load(open(path%'train','r'))
            image_filenames = self.image_filenames[0:10]
        elif split in self.train_split_zeroshot:
            image_filenames = json.load(open(path%split.split('_')[0],'r'))
        else:
            image_filenames = json.load(open(path%split,'r'))

        return image_filenames


    def load_data_ram(self, im_id, key):
        if key=='images':
            data = self.load_image_disk(im_id)

        elif key=='appearance':
            data = self.load_appearance_disk(im_id)

        elif key=='objectscores':
            data = self.load_objectscores_disk(im_id)

        else:
            print('{} key is not recognized'.format(key))

        return data


    def image_filename(self, im_id):
        return self.db[im_id]['filename']


    def load_image_disk(self, im_id):

        filename = self.image_filename(im_id)
        if self.split in ['debug', 'train', 'val', 'trainval'] or self.split in self.train_split_zeroshot:
            im = cv2.imread(osp.join(self.image_dir, 'train2015', filename),1)
        elif self.split=='test':
            im = cv2.imread(osp.join(self.image_dir, 'test2015', filename),1)
        else:
            print('Invalid split')
            return
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB) # Convert BGR to RGB

        return im 


    def load_image(self, im_id, load_disk=False):

        if 'images' in self.store_ram and not load_disk:
            im = self.data_ram['images'][im_id]
        else:        
            im = self.load_image_disk(im_id)    

        return im        



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
                subjname = self.classes.idx2word[sub_cat[j]] # attention can have subject='person' or 'bg'
                #relation = '-'.join(['person',predicate, objname])
                relation = '-'.join([subjname, predicate, objname])
                if relation in self.visualphrases.words():
                    vp_cat = self.visualphrases(relation)
                    labels_visualphrases[j,vp_cat] = 1

        return labels_visualphrases


        
    def load_appearance_disk(self, im_id):

        filepath = osp.join(self.cand_dir, 'appearance_memmap', '%s' + '_' + 'objectappearance_fc7', str(im_id) + '.npy')
        filepath = filepath%'trainval' if (self.split in ['debug', 'train', 'val', 'trainval'] or self.split in self.train_split_zeroshot) else filepath%'test'
        if osp.exists(filepath):
            features_mem = np.memmap(filepath, dtype='float32', mode='r')
            features = np.array(features_mem.reshape(features_mem.shape[0]/1024, 1024))
            del features_mem
        else:
            print('No appearance features loaded for image {}'.format(im_id))
            features = []

        return features


    def load_appearance(self, im_id, cand_id=None, load_disk=False):
        """
        Load appearance feature for (subject, object)
        Input: batch_pair_ids (N,2) [sub_id, obj_id]
                batch_gt (N,) indicator whether groundtruth object or candidate
        Output:
                appearance (N,3,1024) : for subject, object, union boxes
        """

        pair_ids = self.get_pair_ids(im_id, cand_id)
        subject_idx = self.get_obj_id(im_id, idx=pair_ids[:,0])
        object_idx = self.get_obj_id(im_id, idx=pair_ids[:,1])

        appearance_feats = np.zeros((pair_ids.shape[0],2,1024))

        if 'appearance' in self.store_ram and not load_disk:
            features_im = self.data_ram['appearance'][im_id]
            if self.l2norm_input:
                features_im = features_im / np.linalg.norm(features_im, axis=1)[:,None]
        else:
            features_im = self.load_appearance_disk(im_id)
            if self.l2norm_input:
                features_im = features_im / np.linalg.norm(features_im, axis=1)[:,None]

        appearance_feats[:,0,:] = features_im[subject_idx,:]
        appearance_feats[:,1,:] = features_im[object_idx,:]

        return appearance_feats



    def load_objectscores_disk(self, im_id):

        filepath = osp.join(self.cand_dir, 'object_scores_memmap', '%s' + '_' + 'objectscores', str(im_id) + '.npy')
        filepath = filepath%'trainval' if (self.split in ['debug','train', 'val', 'trainval'] or self.split in self.train_split_zeroshot) else filepath%'test'
        if osp.exists(filepath):
            score_mem = np.memmap(filepath, dtype='float32', mode='r')
            scores = np.array(score_mem.reshape(score_mem.shape[0]/81, 81))
            del score_mem
        else:
            scores=[]
        return scores


    def load_objectscores(self, im_id, cand_id, load_disk=False):

        pair_ids = self.get_pair_ids(im_id, cand_id)
        object_scores = np.zeros((pair_ids.shape[0], 2, self.num_classes))
        subject_idx = self.get_obj_id(im_id, idx=pair_ids[:,0])
        object_idx = self.get_obj_id(im_id, idx=pair_ids[:,1])

        if 'objectscores' in self.store_ram and not load_disk:
            scores_im = self.data_ram['objectscores'][im_id]
        else:
            scores_im = self.load_objectscores_disk(im_id)

        object_scores[:,0,:] = scores_im[subject_idx,:]
        object_scores[:,1,:] = scores_im[object_idx,:]

        return object_scores



    """
    Filtering
    """

    def filter_images_noannotations(self):
        '''
        Remove images from image_ids with no relationship annotation
        '''
        self.image_ids_clean = []
        for im_id in self.image_ids:
            if self.db[im_id]['pair_ids'].size >0:
                self.image_ids_clean.append(im_id)
        self.image_ids = self.image_ids_clean



    """
    Get candidates
    """


    def populate_candidates(self):
        """
        Get all candidate pairs from detections (do not filter by object scores at this stage)
        """
        if self.split in ['debug','train', 'val', 'trainval']:
            cand_boxes = json.load(open(self.cand_dir + '/' + 'bbox_hico_trainval_results.json','rb'))
        else:
            cand_boxes = json.load(open(self.cand_dir + '/' + 'bbox_hico_test_results.json' ,'rb'))


        for j in range(len(cand_boxes)):

            im_id = cand_boxes[j]['image_id']

            if im_id not in self.image_ids:
                continue

            obj_id = np.max(self.db[im_id]['obj_id']) + 1 if len(self.db[im_id]['obj_id'])>0 else 0# to keep track of detection index (useful to get back appearance feat after score filtering)
            obj_cat = self.json_category_id_to_contiguous_id[cand_boxes[j]['category_id']] # Attention detectron does not return continous id 
            score = cand_boxes[j]['score']
            width, height = self.image_size(im_id)

            box = cand_boxes[j]['bbox']

            # Transform x,y,w,h -> x,y,x2,y2
            x1, y1 = box[0], box[1]
            x2 = x1 + np.maximum(0., box[2] - 1.)
            y2 = y1 + np.maximum(0., box[3] - 1.)

            # Want boxes to have area at least 1
            w = x2-x1+1
            h = y2-y1+1
            assert w>=1 and h>=1, 'Found candidates of area <1'

            # Get GT labels for subject/object boxes (could be used eventually to refine detections on this dataset)
            obj_gt_class = 0
            is_gt_objects = (self.db[im_id]['is_gt']==1)            
            gt_boxes = self.db[im_id]['boxes'][is_gt_objects,:]
            gt_classes = self.db[im_id]['obj_classes'][is_gt_objects]
            gt_labels_sr = self.db[im_id]['labels_sr'][is_gt_objects,:]
            gt_labels_ro = self.db[im_id]['labels_ro'][is_gt_objects,:]

            # Pre-init labels_sr, labels_ro to background (if no intersection with GT)
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


            # Overlap with all GT boxes
            if len(gt_boxes)>0:
                ovl_gt = get_overlap(gt_boxes, np.array([x1,y1,x2,y2]))
                id_max_ovl = np.argmax(ovl_gt)

                # Label the box as positive for the GT with max overlap, providing that this overlap is above 0.5 
                if ovl_gt[id_max_ovl]>0.5:
                    obj_gt_class  = gt_classes[id_max_ovl]
                    obj_labels_sr = gt_labels_sr[id_max_ovl,:].toarray()
                    obj_labels_ro = gt_labels_ro[id_max_ovl,:].toarray()


            # Append in database
            self.db[im_id]['boxes']          = np.vstack((self.db[im_id]['boxes'], np.array(list([x1,y1,x2,y2])) ))
            self.db[im_id]['obj_classes']    = np.hstack((self.db[im_id]['obj_classes'], np.array([obj_cat])))
            self.db[im_id]['obj_gt_classes'] = np.hstack((self.db[im_id]['obj_gt_classes'], np.array([obj_gt_class])))
            self.db[im_id]['obj_scores']     = np.hstack((self.db[im_id]['obj_scores'], np.array([score])))
            self.db[im_id]['is_gt']          = np.hstack((self.db[im_id]['is_gt'], np.zeros((1), dtype=np.bool)))
            self.db[im_id]['obj_id']         = np.hstack((self.db[im_id]['obj_id'], np.array([obj_id], dtype=np.int32)))
            self.db[im_id]['labels_sr']      = lil_matrix(np.vstack((self.db[im_id]['labels_sr'].toarray(), obj_labels_sr)))
            self.db[im_id]['labels_ro']      = lil_matrix(np.vstack((self.db[im_id]['labels_ro'].toarray(), obj_labels_ro)))


    def label_candidates(self):

        # Test : storing labels in scipy sparse matrix
        for im_id in self.db.keys():

            # All objects in image
            boxes       = self.db[im_id]['boxes']
            obj_classes = self.db[im_id]['obj_classes']
            is_gt       = self.db[im_id]['is_gt']

            idx_cand = np.where(is_gt==0)[0]
            idx_gt   = np.where(is_gt==1)[0]

            if len(idx_cand)==0 or len(idx_gt)==0:
                continue
            
            assert np.max(idx_gt) < np.min(idx_cand), 'Warning db not in order'
            assert np.all(self.db[im_id]['is_gt_pair']==1), 'Warning some pair not GT'

            # Get the groundtruth annotations for this image
            is_gt_pair      = self.db[im_id]['is_gt_pair']
            gt_pair_ids     = self.db[im_id]['pair_ids']
            gt_pair_labels  = self.db[im_id]['labels_r'].toarray() 
            gt_cand_id      = self.db[im_id]['cand_id']
            pair_iou        = self.db[im_id]['pair_iou']
            current_cand_id = np.max(gt_cand_id)+1 if len(gt_cand_id)>0 else 0

            # Form candidate pairs
            ids_subject        = np.where(np.logical_and(obj_classes==1, is_gt==0))[0] # candidate humans
            ids_object         = np.where(np.logical_and(obj_classes>=1, is_gt==0))[0] # all objects included human, excluding bg
            cand_pair_ids      = np.zeros((len(ids_subject)*len(ids_object),2), dtype=np.int32)
            cand_pair_ids[:,0] = np.repeat(ids_subject, len(ids_object))
            cand_pair_ids[:,1] = np.tile(ids_object, len(ids_subject))

            # Discard candidates where subject==object box
            idx           = np.where(cand_pair_ids[:,0]==cand_pair_ids[:,1])[0] 
            cand_pair_ids = np.delete(cand_pair_ids, idx, 0)

            # Label subject-object relation
            idx_pos_pair                    = np.where(np.sum(gt_pair_labels[:,1:],1)>=1)[0]
            gt_pos_pair_ids                 = gt_pair_ids[idx_pos_pair,:]
            gt_pos_pair_labels              = gt_pair_labels[idx_pos_pair,:]
            cand_pair_labels, cand_pair_iou = self.build_label(cand_pair_ids, gt_pos_pair_ids, gt_pos_pair_labels, boxes, obj_classes, self.iou_pos)

            # Merge candidates with GT
            self.db[im_id]['pair_ids']   = np.vstack((gt_pair_ids, cand_pair_ids))
            self.db[im_id]['labels_r']   = lil_matrix(np.vstack((gt_pair_labels, cand_pair_labels)))
            self.db[im_id]['is_gt_pair'] = np.hstack((is_gt_pair, np.zeros((cand_pair_ids.shape[0]),dtype=np.bool)))
            self.db[im_id]['cand_id']    = np.hstack((gt_cand_id, current_cand_id+np.arange(cand_pair_ids.shape[0], dtype=np.int32) ))
            self.db[im_id]['pair_iou']   = np.vstack((pair_iou, cand_pair_iou)) 


    def build_label(self, cand_pair_ids, gt_pair_ids, gt_pair_labels, boxes, obj_classes, iou_pos):

        cand_pair_labels = np.zeros((len(cand_pair_ids), self.num_predicates)) 
        cand_pair_iou    = np.zeros((len(cand_pair_ids),2))

        ids_subject = cand_pair_ids[:,0]
        ids_object  = cand_pair_ids[:,1]

        # Scan the groundtruth relationships for this image and mark as positives candidates overlapping
        for j in range(gt_pair_ids.shape[0]):
            gt_sub  = gt_pair_ids[j,0]
            gt_obj  = gt_pair_ids[j,1]
            sub_cat = obj_classes[gt_sub]
            assert sub_cat==1, 'Subject should be person class'
            obj_cat     = obj_classes[gt_obj]
            subject_box = boxes[gt_sub,:]
            object_box  = boxes[gt_obj,:]

            # Filter candidates by category: both obj_cat and sub_cat
            idx = np.where(np.logical_and(obj_classes[ids_subject]==sub_cat, obj_classes[ids_object]==obj_cat))[0]
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



    """
    Prepare dataset
    """

    def get_vocab_visualphrases(self, actions):
        """
        Get all relations (action, object)
        """
        relations = Vocabulary()
        for k in range(len(actions)):
            relation     = actions[k]
            predicate    = relation['vname']
            predicate    = ' '.join(predicate.split('_'))
            objname      = relation['nname']
            objname      = ' '.join(objname.split('_'))
            visualphrase = '-'.join(['person', predicate, objname])
            relations.add_word(visualphrase, 'noun-verb-noun')

        return relations


    def get_vocab_predicates(self, visualphrases):
        """
        no_interaction class already included
        """
        predicates = Vocabulary()
        predicates.add_word('no interaction', 'verb')
        for visualphrase in visualphrases.words():
            triplet = visualphrase.split('-')
            predicate = triplet[1]
            if predicate not in predicates.words():
                predicates.add_word(predicate, 'verb')

        return predicates


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



    def _build_db(self, annotations):

        db = {}
        for j in range(len(self.image_ids)):
            if j%1000==0:
                print('Preparing entry (load image size) : {}/{}'.format(j,len(self.image_ids)))
            im_id = self.image_ids[j]
            db[im_id] = {}
            self._prep_db_entry(db[im_id])

            # At least fill up image_filename, width, height. Might not be annotations.
            filename = self.image_filenames[j]
            db[im_id]['filename'] = filename
            if self.split in ['debug','train', 'val', 'trainval']:
                im = cv2.imread(osp.join(self.image_dir, 'train2015', filename),1)
            else:
                im = cv2.imread(osp.join(self.image_dir, 'test2015', filename),1)


            height, width, _ = im.shape 
            db[im_id]['width'] = width
            db[im_id]['height'] = height


        # First pass: get the objects
        print('Adding objects in database...')
        self._add_objects(db, annotations)

        print('Adding relationships in database')
        # Second pass : get the relations
        self._add_relationships(db, annotations)

        return db

    def _add_objects(self, db, annotations):

        # First get all object boxes
        objects = np.empty((0,6)) # [im_id, box, obj_cat]
        print('Parse object annotations...')
        for j in range(len(annotations)):
            im_id = annotations[j]['im_id']

            # Check whether annotated image is in split (e.g. train/val/trainval)
            if im_id not in self.image_ids:
                continue

            action_id = annotations[j]['action_id']-1 # -1 from matlab
            human_box = [x-1 for x in annotations[j]['human_box']]
            object_box = [x-1 for x in annotations[j]['object_box']]


            # Append subject
            objects = np.vstack((objects, [im_id] + human_box + [1]))

            # Append object box
            obj_name = self.actions[action_id]['nname']
            obj_name = ' '.join(obj_name.split('_'))
            obj_cat = self.classes(obj_name)
            objects = np.vstack((objects, [im_id] + object_box + [obj_cat])) 
            
        # Get unique objects (unique rows) and fill db
        unique_objects = np.unique(objects, axis=0)

        # Want boxes to have area at least 1
        keep = filter_small_boxes(unique_objects[:,1:5], 1)
        assert len(keep)==unique_objects.shape[0], "Found object boxes of area less than 1"
        

        images = np.unique(unique_objects[:,0])
        print('Populate db objects...')
        for im_id in images:
            idx = np.where(unique_objects[:,0]==im_id)[0] 
            db[im_id]['boxes']          = unique_objects[idx,1:5]
            db[im_id]['obj_classes']    = unique_objects[idx,5].astype(int)
            db[im_id]['obj_gt_classes'] = np.array(unique_objects[idx,5]).astype(int)
            db[im_id]['obj_scores']     = np.ones(len(idx))
            db[im_id]['is_gt']          = np.ones(len(idx), dtype=np.bool)
            db[im_id]['obj_id']         = np.arange(len(idx), dtype=np.int32)


    def _prep_db_entry(self, entry):
        entry['filename'] = None
        entry['width']    = None
        entry['height']   = None
        entry['boxes']          = np.empty((0, 4), dtype=np.float32)
        entry['obj_classes']    = np.empty((0), dtype=np.int32) # will store the detected classes (with object detector)
        entry['obj_gt_classes'] = np.empty((0), dtype=np.int32) # store the GT classes
        entry['obj_scores']     = np.empty((0), dtype=np.float32) # Later: for detections can take the scores over all classes
        entry['is_gt']          = np.empty((0), dtype=np.bool)
        entry['obj_id']         = np.empty((0), dtype=np.int32) # contrary to ann_id, obj_id stores the object id in image (need this because objects get filtered)
        entry['pair_ids']       = np.empty((0,2), dtype=np.int32)
        entry['labels_r']       = lil_matrix((0, self.num_predicates))
        entry['labels_sr']      = lil_matrix((0, len(self.subjectpredicates))) # labels sr attached to subject box: is this box involved in a relation as subject ? 
        entry['labels_ro']      = lil_matrix((0, len(self.objectpredicates))) # labels ro attached to object box: is this box involved in a relation as object ?
        entry['is_gt_pair']     = np.empty((0), dtype=np.bool)
        entry['cand_id']        = np.empty((0), dtype=np.int32) # To identify candidate relation (relative indexing in image)
        entry['pair_iou']       = np.empty((0,2), dtype=np.float32) # IoU with positive GT pairs of subject and object box. Can be use to sample different type of negative candidates


    def _init_coco(self):
        category_ids = self.COCO.getCatIds()
        categories = [c['name'] for c in self.COCO.loadCats(category_ids)]
        self.category_to_id_map = dict(zip(categories, category_ids))
        # Vocabulary of objects
        self.classes = Vocabulary()
        self.classes.add_word('background', 'noun')
        for cat in categories:
            self.classes.add_word(cat, 'noun')
        self.num_classes = len(self.classes)
        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(self.COCO.getCatIds())}
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()}


    def _add_relationships(self, db, annotations):

        # Build all relationships over all images
        all_relationships = np.empty((0,4)) # [im_id, sub_id, obj_id, rel_cat]
        print('Parse relationships annotation...')
        for j in range(len(annotations)):
            im_id = annotations[j]['im_id']

            if im_id not in self.image_ids:
                continue

            action_id  = annotations[j]['action_id']-1 # index -1 from matlab
            human_box  = [x-1 for x in annotations[j]['human_box']]
            object_box = [x-1 for x in annotations[j]['object_box']]

            # Get predicate, obj_cat
            predicate_name = self.actions[action_id]['vname']
            predicate_name = ' '.join(predicate_name.split('_'))
            rel_cat  = self.predicates(predicate_name)
            obj_name = self.actions[action_id]['nname']
            obj_name = ' '.join(obj_name.split('_'))
            obj_cat  = self.classes(obj_name)
            sub_cat  = 1

            # Get sub_id, obj_id
            boxes   = db[im_id]['boxes']
            classes = db[im_id]['obj_classes']

            sub_id = np.where(np.logical_and(np.all(boxes==human_box, axis=1), classes==sub_cat))[0]
            obj_id = np.where(np.logical_and(np.all(boxes==object_box, axis=1), classes==obj_cat))[0]

            # Append in relationships
            all_relationships = np.vstack((all_relationships, np.array([im_id, sub_id, obj_id, rel_cat])))

        # Fill database
        print('Populate db relationships...') 
        for im_id in self.image_ids:
            idx = np.where(all_relationships[:,0]==im_id)[0]

            if len(idx)==0:
                continue

            # Fill with positives
            relationships_im        = all_relationships[idx,1:]
            relationships_unique    = multilabel_transform(relationships_im, self.num_predicates) # Remove duplicates + binarize
            db[im_id]['pair_ids']   = relationships_unique[:,:2].astype(np.int32)
            db[im_id]['labels_r']   = lil_matrix(relationships_unique[:,2:])
            db[im_id]['is_gt_pair'] = np.ones((relationships_unique.shape[0]), dtype=np.bool)
            db[im_id]['cand_id']    = np.arange(relationships_unique.shape[0], dtype=np.int32)
            db[im_id]['pair_iou']   = np.ones((relationships_unique.shape[0],2), dtype=np.float32) # Iou of positive is 1 !


            # Multilabel: solve issue duplicate pairs (pairs that overlap >0.7)
            iou_pos = 0.7
            labels_r_multilabel, _ = self.build_label(db[im_id]['pair_ids'], db[im_id]['pair_ids'], \
                                                    db[im_id]['labels_r'].toarray(), \
                                                    db[im_id]['boxes'], db[im_id]['obj_classes'], iou_pos) 

            db[im_id]['labels_r'] = lil_matrix(labels_r_multilabel)


            # Add (human, object) negative pairs
            if self.neg_GT:
                obj_classes    = db[im_id]['obj_classes']
                sub_id         = np.where(obj_classes==1)[0] # humans
                obj_id         = np.where(obj_classes>=1)[0] # objects (included human)
                all_pairs      = np.zeros((len(sub_id)*len(obj_id),2), dtype=np.int32)
                all_pairs[:,0] = np.repeat(sub_id, len(obj_id))
                all_pairs[:,1] = np.tile(obj_id, len(sub_id))
                is_pos = []
                for j in range(relationships_unique.shape[0]):
                    idx = np.where(np.logical_and((all_pairs[:,0]==relationships_unique[j,0]), (all_pairs[:,1]==relationships_unique[j,1])) >0)[0]
                    if len(idx)>0:
                        is_pos.append(idx[0])
                is_neg    = np.setdiff1d(np.arange(all_pairs.shape[0]), is_pos)
                neg_pairs = all_pairs[is_neg,:]

                idx = np.where(neg_pairs[:,0]==neg_pairs[:,1])[0] # Discard candidates where subject==object box
                neg_pairs = np.delete(neg_pairs, idx, 0)

                gt_indicator    = np.ones((neg_pairs.shape[0]), np.bool)
                cand_id_current = np.max(db[im_id]['cand_id']) + 1 if len(db[im_id]['cand_id'])>0 else 0 

                db[im_id]['pair_ids']   = np.vstack((db[im_id]['pair_ids'], neg_pairs))
                db[im_id]['is_gt_pair'] = np.hstack((db[im_id]['is_gt_pair'], gt_indicator)) # it's not a gt pair, but it's made of gt boxes... 
                db[im_id]['cand_id']    = np.hstack((db[im_id]['cand_id'], cand_id_current + np.arange(neg_pairs.shape[0], dtype=np.int32)))


                # Labels the negative pairs

                # Some of these negative pairs intersect a gt: label them !!
                iou_pos = 0.5 
                idx_pos_pair = np.where(np.sum(db[im_id]['labels_r'][:,1:],1)>=1)[0]
                neg_labels, neg_iou = self.build_label(neg_pairs, db[im_id]['pair_ids'][idx_pos_pair], \
                                                    db[im_id]['labels_r'][idx_pos_pair,:].toarray(), \
                                                    db[im_id]['boxes'], db[im_id]['obj_classes'], iou_pos)

                db[im_id]['labels_r'] = lil_matrix(np.vstack((db[im_id]['labels_r'].toarray(), neg_labels)))
                db[im_id]['pair_iou'] = np.vstack((db[im_id]['pair_iou'], neg_iou))


            # Get bigram labels_sr, labels_ro for each object -> these labels are attached to objects
            objects_ids = db[im_id]['obj_id']

            for o in range(len(objects_ids)):
                obj_id  = objects_ids[o]
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
                            relation  = '-'.join([objname, predicate])
                            ind_sr    = self.subjectpredicates(relation)
                            labels_sr[0, ind_sr] = 1

                    # If no label, label as no_interaction
                    if np.sum(labels_sr)==0:
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
                    if len(ind_rels)>0:
                        for r in ind_rels:
                            predicate = self.predicates.idx2word[r+1]
                            relation = '-'.join([predicate, objname])
                            ind_ro = self.objectpredicates(relation)
                            labels_ro[0, ind_ro] = 1

                    if np.sum(labels_ro)==0:    
                        # Label as no interaction
                        relation = '-'.join(['no interaction', objname])
                        ind_ro = self.objectpredicates(relation)
                        labels_ro[0, ind_ro] = 1

                db[im_id]['labels_ro'] = lil_matrix(np.vstack((db[im_id]['labels_ro'].toarray(), labels_ro))) 


    def get_occurrences(self, split):
        """
        Scan the cand_positives to get the occurrences -> number of positive candidates <> number of positives annotated (because of duplicate boxes)
        """
        cand_positives = pickle.load(open(osp.join(self.data_dir, 'cand_positives_' + split + '.pkl'),'rb'))

        occurrences = {tripletname:0 for tripletname in self.vocab_grams['sro'].words()}
        for j in range(cand_positives.shape[0]):
            im_id = cand_positives[j,0]
            cand_id = cand_positives[j,1]
            triplet_cats = np.where(self.get_labels_visualphrases(im_id, cand_id))[1]
            for _,triplet_cat in enumerate(triplet_cats):
                tripletname = self.vocab_grams['sro'].idx2word[triplet_cat]
                occurrences[tripletname] += 1

        return occurrences


    def get_occurrences_precomp(self, split, word_type='triplet'):
        """ Get number of triplets annotated in split """

        triplets_remove = []
        if split in self.train_split_zeroshot:
            split, zeroshotset = split.split('_')
            triplets_remove = pickle.load(open(osp.join(self.data_dir, 'zeroshottriplets.pkl'), 'rb'))

        filename = osp.join(self.data_dir, 'occurrences.csv')
        count = 0
        occurrences = {}
        with open(filename) as f:
            reader = csv.DictReader(f)
            for line in reader:
                occ_split = line['occ_' + split]
                action_name = line['action_name']
                triplet_name = self.vocab_grams['sro'].idx2word[count]
                if triplet_name in triplets_remove:
                    occurrences[triplet_name] = 0
                else:
                    occurrences[triplet_name] = int(occ_split)
                count += 1

        return occurrences


    def get_zeroshottriplets(self):

        triplets_remove= [  'person-hold-elephant',\
                            'person-pet-cat',\
                            'person-watch-giraffe',\
                            'person-herd-cow',\
                            'person-ride-horse',\
                            'person-walk-sheep',\
                            'person-hug-dog',\
                            'person-eat-banana',\
                            'person-hold-carrot',\
                            'person-carry-hot dog',\
                            'person-eat-donut',\
                            'person-pick up-cake',\
                            'person-carry-skateboard',\
                            'person-hold-surfboard',\
                            'person-jump-snowboard',\
                            'person-ride-skis',\
                            'person-straddle-motorcycle',\
                            'person-inspect-bicycle',\
                            'person-lie on-bed',\
                            'person-hold-wine glass',\
                            'person-carry-bottle',\
                            'person-hold-knife',\
                            'person-throw-frisbee',\
                            'person-sit on-bench',\
                            'person-wear-backpack']

        return triplets_remove


