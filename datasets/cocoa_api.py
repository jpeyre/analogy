import __init__
import os.path as osp
import os, json
import copy
import numpy as np
import cv2
import pdb
import scipy.misc
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from utils import multilabel_transform, get_overlap, filter_small_boxes, Vocabulary
from scipy.sparse import lil_matrix
import scipy.sparse as sparse
import numbers
import cPickle as pickle
from Dataset import BaseDataset
import csv

class Cocoa(BaseDataset):

    def __init__(self, data_dir, image_dir, split, cand_dir, thresh_file=None, use_gt=False, add_gt=True, train_mode=True, jittering=False, filter_images=True, nms_thresh=0.5, store_ram=[], l2norm_input=False, neg_GT=True):
        super(Cocoa, self).__init__()

        self.data_name = 'cocoa'
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
        self.coco_ann_dir = '/sequoia/data2/jpeyre/datasets/coco/annotations'
        self.l2norm_input = l2norm_input

        self.neg_GT = True # whether to form negative pairs from GT at training or not
        self.iou_pos = 0.5
        self.iou_neg = 0.5

        self.d_appearance = 1024

        # Base split (for zero-shot)
        base_split = self.split.split('_')[0]

        # Filter detections (per-class threshold to maintain precision 0.3 measured on COCO dataset)
        if self.thresh_file:
            self.dets_thresh = np.load(osp.join(self.cand_dir, self.thresh_file + '.npy'))
        else:
            self.dets_thresh = None


        """ Load cocoa annotations """
        with open("{0}/{1}".format(self.data_dir, 'cocoa_beta2015.json')) as f:
            cocoa = json.load(f)

        # annotations with agreement of at least 1 mturk annotator
        cocoa_1 = cocoa['annotations']['1']
        # annotations with agreement of at least 2 mturk annotator
        cocoa_2 = cocoa['annotations']['2']
        # annotations with agreement of at least 3 mturk annotator
        cocoa_3 = cocoa['annotations']['3']

        """ Merge cocoa_1, cocoa_2, cocoa_3 """
        self.annotations = cocoa_1 + cocoa_2 + cocoa_3

        """ Load visual verb net """
        with open("{0}/{1}".format(self.data_dir, 'visual_verbnet_beta2015.json')) as f:
            vvn = json.load(f)

        # list of 145 visual actions contained in VVN
        visual_actions = vvn['visual_actions']
        # list of 17 visual adverbs contained in VVN
        visual_adverbs = vvn['visual_adverbs'] 

        """ load coco annotations """
        ANN_FILE_PATH = "{0}/instances_{1}.json".format(self.coco_ann_dir,'train2014')

        self.COCO = COCO( ANN_FILE_PATH )
        self._init_coco()

        """ Get all image ids in Coco-a """
        # Print list of all images in images.ids
        all_image_ids = []
        for annot in self.annotations:
            im_id = annot['image_id']
            if im_id not in all_image_ids:
                all_image_ids.append(im_id)
        all_image_ids = np.array(all_image_ids)
                
        print('There are {} images in COCO-a'.format(len(all_image_ids)))
        if not osp.exists(osp.join(self.data_dir, 'image.ids')):
            np.savetxt(osp.join('/sequoia/data2/jpeyre/datasets/cocoa', 'images.ids'), all_image_ids, fmt='%d') 


        """ Get the vocab of predicates and visualphrases """
        self.predicates, self.id_to_action = self.get_vocab_actions(visual_actions) # the vocabulary of all visual actions
        self.visualphrases, self.occ_triplets, self.subjectpredicates, self.objectpredicates = self.get_vocab_visualphrases(self.annotations)
        self.num_visualphrases = len(self.visualphrases)


        """ Pre-proc """
        self.create_splits(all_image_ids)
 
        """ Image ids """
        self.image_ids = self.get_image_ids(base_split)


        """ Database """
        if osp.exists(osp.join(self.data_dir, 'db_' + base_split + '.pkl')):
            self.db = pickle.load(open(osp.join(self.data_dir, 'db_' + base_split + '.pkl'),'rb'))
        else:
            # Do not build db for zeroshot
            assert base_split==self.split, 'Attention we do not build db for zero-shot'
            self.db = self._build_db(self.annotations)
            self.populate_candidates()
            self.label_candidates()
            pickle.dump(self.db, open(osp.join(self.data_dir, 'db_' + self.split + '.pkl'),'wb'))


        # Vocab wrapper
        self.vocab = self.build_vocab(self.classes, self.predicates)
        # Save vocab to compute word embeddings: WARNING: we can have duplicate word depending on POS. Choice to allow training different word embeddings. 
        pickle.dump(self.vocab.idx2word.values(), open(osp.join(self.data_dir, 'vocab' + '.pkl'), 'wb'))

        
        self.vocab_grams = {'s':self.classes,
                            'o':self.classes,
                            'r':self.predicates,
                            #'sr':self.subjectpredicates,
                            #'ro':self.objectpredicates,
                            'sr':[],
                            'ro':[],
                            'sro':self.visualphrases,
                            'all':self.vocab}

        self.idx_sro_to = self.get_idx_between_vocab(self.vocab_grams['sro'], self.vocab_grams)
        self.idx_to_vocab = self.get_idx_in_vocab(self.vocab_grams, self.vocab_grams['all']) # get idx of vocab_grams in vocab_all (to access pre-computed word embeddings)

        # Pre-trained word embeddings for subject/object/verb
        self.word_embeddings = pickle.load(open(osp.join(self.data_dir, 'pretrained_embeddings_w2v.pkl'), 'rb'))
      


        # ATTENTION !!!!!!!!!!!!!!!!
        #print('ATTENTION !!!!!!!!!!!!! we use random word2vec embedding')
        #self.word_embeddings = pickle.load(open(osp.join(self.data_dir, 'random_embeddings.pkl'), 'rb'))




        if self.l2norm_input:
            if (np.linalg.norm(self.word_embeddings,axis=1)==0).any():
                raise Exception('At least one word embedding vector is 0 (would cause nan after normalization)')
            self.word_embeddings = self.word_embeddings / np.linalg.norm(self.word_embeddings,axis=1)[:,None]
 


 
        # Load candidates for training
        if train_mode:
            if osp.exists(osp.join(self.data_dir, 'cand_positives_' + self.split + '.pkl')):
                self.cand_positives = pickle.load(open(osp.join(self.data_dir, 'cand_positives_' + self.split + '.pkl'),'rb'))
                self.cand_negatives = pickle.load(open(osp.join(self.data_dir, 'cand_negatives_' + self.split + '.pkl'),'rb'))
            else:
                self.cand_positives, self.cand_negatives = self.get_training_candidates(use_gt=self.use_gt, add_gt=self.add_gt, thresh_file=self.thresh_file)
                pickle.dump(self.cand_positives, open(osp.join(self.data_dir, 'cand_positives_' + self.split + '.pkl'),'wb'))
                pickle.dump(self.cand_negatives, open(osp.join(self.data_dir, 'cand_negatives_' + self.split + '.pkl'),'wb'))
        else:
            self.candidates = self.get_test_candidates(use_gt=self.use_gt, thresh_file=self.thresh_file, nms_thresh=self.nms_thresh)

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
        return self.db[im_id]['file_name']


    def load_image_disk(self, im_id):

        filename = self.image_filename(im_id)
        im = cv2.imread(osp.join(self.image_dir, 'train2014', filename),1) # all images or from train2014 split
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

        # Return visual phrases labels
        labels_visualphrases = np.zeros((pair_ids.shape[0],self.num_visualphrases))
        for j in range(pair_ids.shape[0]):
            ind_rels = np.where(labels_predicates[j,:]==1)[0]
            for r in ind_rels:
                predicate = self.predicates.idx2word[r]
                objname = self.classes.idx2word[obj_cat[j]]
                relation = '-'.join(['person',predicate, objname])
                if relation in self.visualphrases.words():
                    vp_cat = self.visualphrases(relation)
                    labels_visualphrases[j,vp_cat] = 1

        return labels_visualphrases


    def get_image_ids(self, split):

        assert split in ['train','val','trainval','test', 'all', 'debug'], 'Invalid split'

        if split=='all':
            image_ids = np.loadtxt(open(osp.join(self.data_dir, 'images.ids'),'r')).astype(np.int32)
        else:
            image_ids = np.loadtxt(open(osp.join(self.data_dir, split + '.ids'),'r')).astype(np.int32)

        return image_ids



    def get_vocab_actions(self, visual_actions):
        actions = Vocabulary()
        actions.add_word('no interaction', 'verb')

        # Correspondency with annotation id
        id_to_action = {}
        for visual_action in visual_actions:    
            actioname = visual_action['name']
            actioname = ' '.join(actioname.split('_'))
            actions.add_word(actioname, 'verb')
            id_to_action[visual_action['id']] = actioname


        # Write vocab of visual phrases in file
        with open(osp.join(self.data_dir, 'vocab_predicates.csv'), 'wb') as csvfile:
            predicate_writer = csv.writer(csvfile)
            for predicate in actions.words():
                predicate_writer.writerow([predicate])

        return actions, id_to_action


    def get_vocab_visualphrases(self, annotations):
        """ Vocabulary of visual phrases and bigrams """

        visual_phrases = Vocabulary()
        subjectpredicates = Vocabulary()
        objectpredicates = Vocabulary()
        occ_triplets = {}

        subjectname = 'person'

        for annot in annotations:
        
            actions_id = annot['visual_actions']
            object_id = annot['object_id']

            # If no object, only append as a bigram subject-predicate
            if object_id==-1:
                for action_id in actions_id:
                    predicate = self.id_to_action[action_id]
                    bigram_sr = '-'.join([subjectname, predicate])    
                    if bigram_sr not in subjectpredicates.words():
                        subjectpredicates.add_word(bigram_sr, 'noun-verb')
                continue

            # Else there is an interacting object
            object_anns  = self.COCO.loadAnns(object_id)[0]
            objectname   = self.COCO.cats[object_anns['category_id']]['name']    
            
            for action_id in actions_id:
                predicate = self.id_to_action[action_id]
               
                bigram_sr = '-'.join([subjectname, predicate]) 
                bigram_ro = '-'.join([predicate, objectname])
                triplet = '-'.join([subjectname, predicate, objectname])
                
                if triplet not in visual_phrases.words():
                    visual_phrases.add_word(triplet, 'noun-verb-noun')
                    occ_triplets[triplet] = 0

                if bigram_sr not in subjectpredicates.words():
                    subjectpredicates.add_word(bigram_sr, 'noun-verb')

                if bigram_ro not in objectpredicates.words():
                    objectpredicates.add_word(bigram_ro, 'verb-noun')
                    
                occ_triplets[triplet] +=1 


        # Write vocab of visual phrases in file
        with open(osp.join(self.data_dir, 'vocab_visualphrases.csv'), 'wb') as csvfile:
            vp_writer = csv.writer(csvfile)
            for vp in visual_phrases.words():
                vp_writer.writerow([vp])


        return visual_phrases, occ_triplets, subjectpredicates, objectpredicates


    def load_appearance_disk(self, im_id):
        # Keeping features for all images in the same directory
        filepath = osp.join(self.cand_dir, 'appearance_memmap', str(im_id) + '.npy')
        if osp.exists(filepath):
            features_mem = np.memmap(filepath, dtype='float32', mode='r')
            features = np.array(features_mem.reshape(features_mem.shape[0]/self.d_appearance, self.d_appearance))
            del features_mem
        else:
            features = []

        return features


    def load_appearance(self, im_id, cand_id=None, load_disk=False):
        """
        Load appearance feature for (subject, object)
        """

        pair_ids = self.get_pair_ids(im_id, cand_id)
        subject_idx = self.get_obj_id(im_id, idx=pair_ids[:,0])
        object_idx = self.get_obj_id(im_id, idx=pair_ids[:,1])

        appearance_feats = np.zeros((pair_ids.shape[0],2,self.d_appearance))

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

        filepath = osp.join(self.cand_dir, 'object_scores_memmap', str(im_id) + '.npy')
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

    def filter_pairs_salient(self, im_id, idx=None):
        """
        For HICO: all pairs are salient
        """
        if idx is None:
            idx = self.db[im_id]['pair_ids'][:,0]
        else:
            if isinstance(idx, numbers.Number):
                idx = np.array([idx])
        return idx



    """
    Pre-processing functions : only called once 
    """

    def get_occurrences(self, split):
        return []


    def _init_coco(self):
        category_ids = self.COCO.getCatIds()
        categories = [c['name'] for c in self.COCO.loadCats(category_ids)]
        self.category_to_id_map = dict(zip(categories, category_ids))
        self.classes = Vocabulary()
        self.classes.add_word('background', 'noun')
        for cat in categories:
            self.classes.add_word(cat, 'noun')
        self.num_classes = len(self.classes)
        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(self.COCO.getCatIds())}
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()}


    def _prep_db_entry(self, entry):

        entry['boxes'] = np.empty((0, 4), dtype=np.float32)             # coordinates of candidate boxes
        entry['obj_classes'] = np.empty((0), dtype=np.int32)            # detected classes (by object detector)
        entry['obj_scores'] = np.empty((0), dtype=np.float32) # Later: for detections can take the scores over all classes
        entry['obj_gt_classes'] = np.empty((0), dtype=np.int32)         # groundtruth classes
        entry['is_gt'] = np.empty((0), dtype=np.bool)                   # whether the box is groundtruth or from object detector
        entry['ann_id'] = np.empty((0), dtype=np.int64) # ann_id for objects in COCO annotations. Careful incode it on int64 because large indices 
        entry['obj_id'] = np.empty((0), dtype=np.int32) # contrary to ann_id, obj_id stores the object id in image (need this because objects get filtered)
        entry['pair_ids'] = np.empty((0,2), dtype=np.int32)             # (obj_id subject, obj_id object)
        entry['labels_r'] = lil_matrix((0, len(self.predicates))) # labels for predicates  
        entry['labels_sr'] = lil_matrix((0, len(self.subjectpredicates))) # labels sr attached to subject box: is this box involved in a relation as subject ? 
        entry['labels_ro'] = lil_matrix((0, len(self.objectpredicates))) # labels ro attached to object box: is this box involved in a relation as object ?
        entry['is_gt_pair'] = np.empty((0), dtype=np.bool)              # whether the pair of boxes is from groundtruth or not
        entry['cand_id'] = np.empty((0), dtype=np.int32)                # candidate id for pair of boxes (relative indexing in image)
        #entry['cand_id_cocoa'] = np.empty((0), dtype=np.int32)          # candidate id for pair of boxes by cocoa


    def _build_db(self, annotations):

        db = {}

        imgs_info = copy.deepcopy(self.COCO.loadImgs(self.image_ids.tolist()))

        """ Copy image info from COCO """
        for i in range(len(imgs_info)):
            im_id = imgs_info[i]['id']
            db[im_id] = imgs_info[i]
            self._prep_db_entry(db[im_id])

        """ First pass: get the objects """
        print('Adding objects in database...')
        #self._add_gt_objects(db, annotations)
        self._add_all_gt_objects(db, annotations)

        """ Second pass: get the relationships """
        print('Adding relationships in database')
        self._add_gt_relationships(db, annotations)

        return db


    def _add_all_gt_objects(self, db, annotations):
        """
        Add ALL the GT objects in COCO dataset (not only those occuring in relations) in order of their ann_id
        As done for V-COCO
        """
        imgs_info = copy.deepcopy(self.COCO.loadImgs(self.image_ids.tolist()))

        for img_info in imgs_info:
            im_id = img_info['id']

            if im_id not in self.image_ids:
                continue

            ann_ids = self.COCO.getAnnIds(imgIds=img_info['id'], iscrowd=None)
            objs = self.COCO.loadAnns(ann_ids)

            # Sanitize bboxes -- some are invalid
            valid_objs = []
            valid_ann_ids = []
            width = img_info['width']
            height = img_info['height']
            for i, obj in enumerate(objs):
              if 'ignore' in obj and obj['ignore'] == 1:
                  continue
              # Convert form x1, y1, w, h to x1, y1, x2, y2
              x1 = obj['bbox'][0]
              y1 = obj['bbox'][1]
              x2 = x1 + np.maximum(0., obj['bbox'][2] - 1.)
              y2 = y1 + np.maximum(0., obj['bbox'][3] - 1.)
              x1, y1, x2, y2 = self.clip_xyxy_to_image(
                  x1, y1, x2, y2, height, width)
              # Require non-zero seg area and more than 1x1 box size
              if obj['area'] > 0 and x2 > x1 and y2 > y1:
                obj['clean_bbox'] = [x1, y1, x2, y2]
                valid_objs.append(obj)
                valid_ann_ids.append(ann_ids[i])
            num_valid_objs = len(valid_objs)
            assert num_valid_objs == len(valid_ann_ids)

            boxes = np.zeros((num_valid_objs, 4))
            gt_classes = np.zeros((num_valid_objs), dtype=np.int32)
            ann_id = np.zeros((num_valid_objs), dtype=np.int64) # ann_id for objects in COCO annotations 
            labels_sr = np.zeros((num_valid_objs, len(self.subjectpredicates))) # to do: check subjectpredicates is same vocab as actions (same order) but shifted by 1 because of no_interaction class
            labels_ro = np.zeros((num_valid_objs, len(self.objectpredicates))) # to do: check subjectpredicates is same vocab as actions (same order) but shifted by 1 because of no_interaction class

            for ix, obj in enumerate(valid_objs):
              cls = self.json_category_id_to_contiguous_id[obj['category_id']]
              boxes[ix, :] = obj['clean_bbox']
              gt_classes[ix] = cls
              ann_id[ix] = obj['id']


            db[im_id]['ann_id'] = np.append(db[im_id]['ann_id'], ann_id)
            db[im_id]['boxes'] = np.append(db[im_id]['boxes'], boxes, axis=0)
            db[im_id]['obj_classes'] = np.append(db[im_id]['obj_classes'], gt_classes)
            db[im_id]['obj_scores'] = np.append(db[im_id]['obj_scores'], np.ones(len(valid_objs), dtype=np.float32))
            db[im_id]['obj_gt_classes'] = np.append(db[im_id]['obj_gt_classes'], gt_classes)
            db[im_id]['is_gt'] = np.append(db[im_id]['is_gt'], np.ones(len(valid_objs), dtype=np.bool))
            db[im_id]['obj_id'] = np.append(db[im_id]['obj_id'], np.arange(len(valid_objs), dtype=np.int32))
            db[im_id]['labels_sr'] = lil_matrix(labels_sr)
            db[im_id]['labels_ro'] = lil_matrix(labels_ro)


        # Now we fill labels_sr, labels_ro
        for annot in annotations:
            
            im_id = annot['image_id']

            if im_id not in self.image_ids:
                continue

            # Get visual actions to fill labels_sr
            visual_actions = annot['visual_actions']

            subject_id_coco = annot['subject_id']
            object_id_coco = annot['object_id']

            # Get annotations in COCO
            subject_anns = self.COCO.loadAnns(subject_id_coco)[0]
            subjectname  = self.COCO.cats[subject_anns['category_id']]['name']
            sub_cat      = self.classes.word2idx[subjectname]


            # Fill labels_sr with the new actions
            idx = np.where(db[im_id]['ann_id']==subject_id_coco)[0][0]
            labels_sr = db[im_id]['labels_sr'].toarray()
            for action_id in visual_actions:
                predicate = self.id_to_action[action_id]
                relation = '-'.join(['person',predicate])
                ind_sr = self.subjectpredicates(relation)
                labels_sr[idx, ind_sr] = 1
            db[im_id]['labels_sr'] = lil_matrix(labels_sr)


            # Fill labels_ro 
            if object_id_coco==-1:
                continue

            object_anns  = self.COCO.loadAnns(object_id_coco)[0]
            objectname   = self.COCO.cats[object_anns['category_id']]['name']
            obj_cat      = self.classes.word2idx[objectname]

            idx = np.where(db[im_id]['ann_id']==object_id_coco)[0][0]
            labels_ro = db[im_id]['labels_ro'].toarray()
            for action_id in visual_actions:
                predicate = self.id_to_action[action_id]
                relation = '-'.join([predicate, objectname])
                ind_ro = self.objectpredicates(relation)
                labels_ro[idx, ind_ro] = 1
            db[im_id]['labels_ro'] = lil_matrix(labels_ro)



    def _add_gt_objects(self, db, annotations):
        """
        Add GT objects in order of their ann_id -> appearance features will be saved this order
        """

        objects = np.empty((0,6)) # [im_id, box, obj_cat]
        id_done = []
        for annot in annotations:

            im_id = annot['image_id']

            if im_id not in self.image_ids:
                continue

            # Get visual actions to fill labels_sr
            visual_actions = annot['visual_actions']

            width = db[im_id]['width']
            height = db[im_id]['height']

            subject_id_coco = annot['subject_id']
            object_id_coco = annot['object_id']
            
            # Get annotations in COCO
            subject_anns = self.COCO.loadAnns(subject_id_coco)[0]
            subjectname  = self.COCO.cats[subject_anns['category_id']]['name']
            sub_cat      = self.classes.word2idx[subjectname]

            # Convert form x1, y1, w, h to x1, y1, x2, y2
            x1 = subject_anns['bbox'][0]
            y1 = subject_anns['bbox'][1]
            x2 = x1 + np.maximum(0., subject_anns['bbox'][2] - 1.)
            y2 = y1 + np.maximum(0., subject_anns['bbox'][3] - 1.)

            # TMP: how often does it clip? Want to check whether box in right sense x/y
            x1, y1, x2, y2 = self.clip_xyxy_to_image(x1, y1, x2, y2, height, width)

            # Require non-zero seg area and more than 1x1 box size
            if subject_anns['area'] > 0 and x2 > x1 and y2 > y1:

                if subject_id_coco not in id_done:

                    db[im_id]['ann_id'] = np.hstack((db[im_id]['ann_id'], np.array([subject_id_coco])))
                    db[im_id]['boxes'] = np.vstack((db[im_id]['boxes'], np.array(list([x1,y1,x2,y2]))))
                    db[im_id]['obj_classes'] = np.hstack((db[im_id]['obj_classes'], np.array([sub_cat])))
                    db[im_id]['obj_scores'] = np.hstack((db[im_id]['obj_scores'], np.ones((1), dtype=np.float32)))
                    db[im_id]['obj_gt_classes'] = np.hstack((db[im_id]['obj_gt_classes'], np.array([sub_cat])))
                    db[im_id]['is_gt'] = np.hstack((db[im_id]['is_gt'], np.ones((1), dtype=np.bool)))
                    obj_id = np.max(db[im_id]['obj_id'])+1 if len(db[im_id]['obj_id']>0) else 0
                    db[im_id]['obj_id'] = np.hstack((db[im_id]['obj_id'], np.array([obj_id])))

                    # Init labels_sr and ro
                    db[im_id]['labels_sr'] = lil_matrix(np.vstack((db[im_id]['labels_sr'].toarray(), np.zeros((1,len(self.subjectpredicates))))))
                    db[im_id]['labels_ro'] = lil_matrix(np.vstack((db[im_id]['labels_ro'].toarray(), np.zeros((1,len(self.objectpredicates))))))

                    id_done.append(subject_id_coco)

                # Fill labels_sr with the new actions
                idx = np.where(db[im_id]['ann_id']==subject_id_coco)[0][0]
                labels_sr = db[im_id]['labels_sr'].toarray()
                for action_id in visual_actions:
                    predicate = self.id_to_action[action_id]
                    relation = '-'.join(['person',predicate]) 
                    ind_sr = self.subjectpredicates(relation)
                    labels_sr[idx, ind_sr] = 1
                db[im_id]['labels_sr'] = lil_matrix(labels_sr)


            if object_id_coco==-1:
                continue

            object_anns  = self.COCO.loadAnns(object_id_coco)[0]
            objectname   = self.COCO.cats[object_anns['category_id']]['name']
            obj_cat      = self.classes.word2idx[objectname]

            # Convert form x1, y1, w, h to x1, y1, x2, y2
            x1 = object_anns['bbox'][0]
            y1 = object_anns['bbox'][1]
            x2 = x1 + np.maximum(0., object_anns['bbox'][2] - 1.)
            y2 = y1 + np.maximum(0., object_anns['bbox'][3] - 1.)
            x1, y1, x2, y2 = self.clip_xyxy_to_image(x1, y1, x2, y2, height, width)
            # Require non-zero seg area and more than 1x1 box size
            if object_anns['area'] > 0 and x2 > x1 and y2 > y1:

                if object_id_coco not in id_done:

                    db[im_id]['ann_id'] = np.hstack((db[im_id]['ann_id'], np.array([object_id_coco])))
                    db[im_id]['boxes'] = np.vstack((db[im_id]['boxes'], np.array(list([x1,y1,x2,y2]))))
                    db[im_id]['obj_classes'] = np.hstack((db[im_id]['obj_classes'], np.array([obj_cat])))
                    db[im_id]['obj_scores'] = np.hstack((db[im_id]['obj_scores'], np.ones((1), dtype=np.float32)))
                    db[im_id]['obj_gt_classes'] = np.hstack((db[im_id]['obj_gt_classes'], np.array([obj_cat])))
                    db[im_id]['is_gt'] = np.hstack((db[im_id]['is_gt'], np.ones((1), dtype=np.bool)))
                    obj_id = np.max(db[im_id]['obj_id'])+1 if len(db[im_id]['obj_id']>0) else 0
                    db[im_id]['obj_id'] = np.hstack((db[im_id]['obj_id'], np.array([obj_id])))

                    # Init labels_sr and ro
                    db[im_id]['labels_sr'] = lil_matrix(np.vstack((db[im_id]['labels_sr'].toarray(), np.zeros((1,len(self.subjectpredicates))))))
                    db[im_id]['labels_ro'] = lil_matrix(np.vstack((db[im_id]['labels_ro'].toarray(), np.zeros((1,len(self.objectpredicates))))))

                    id_done.append(object_id_coco)

                # Fill labels_ro with the new actions
                idx = np.where(db[im_id]['ann_id']==object_id_coco)[0][0]
                labels_ro = db[im_id]['labels_ro'].toarray()
                for action_id in visual_actions:
                    predicate = self.id_to_action[action_id]
                    relation = '-'.join([predicate, objectname]) 
                    ind_ro = self.objectpredicates(relation)
                    labels_ro[idx, ind_ro] = 1
                db[im_id]['labels_ro'] = lil_matrix(labels_ro)



    def _add_gt_relationships(self, db, annotations):

        all_relationships = np.empty((0,4)) # [im_id, sub_id, obj_id, rel_cat]


        for annot in annotations:

            im_id = annot['image_id']

            if im_id not in self.image_ids:
                continue

            subject_id_coco = annot['subject_id']
            object_id_coco = annot['object_id']

            if object_id_coco==-1:
                continue

            visual_actions = annot['visual_actions']

            # Attention : merge duplicates if any (could be possible as there are 3 annotations splits in coco-a depending on the agreement of annotators)
            ann_id = db[im_id]['ann_id'] # a box is identified by its ann_id

            sub_id = np.where(ann_id == subject_id_coco)[0]
            obj_id = np.where(ann_id == object_id_coco)[0]

            # The boxes might have been removed if too small
            if len(sub_id)==0 or len(obj_id)==0:
                continue
            else:
                sub_id = sub_id[0]
                obj_id = obj_id[0]

                # Append in relationships
                for action_id in visual_actions:
                    predicate = self.id_to_action[action_id]
                    rel_cat = self.predicates.word2idx[predicate] 
                    all_relationships = np.vstack((all_relationships, np.array([im_id, sub_id, obj_id, rel_cat])))


        # Fill database
        img_no_rel = 0 
        for im_id in self.image_ids:
            idx = np.where(all_relationships[:,0]==im_id)[0]

            if len(idx)==0:
                img_no_rel +=1
                continue

            # Fill with positives
            relationships_im = all_relationships[idx,1:]
            relationships_unique = multilabel_transform(relationships_im, len(self.predicates)) # Remove duplicates + binarize
            db[im_id]['pair_ids'] = relationships_unique[:,:2].astype(np.int32)
            db[im_id]['labels_r'] = lil_matrix(relationships_unique[:,2:])
            db[im_id]['is_gt_pair'] = np.ones((relationships_unique.shape[0]), dtype=np.bool)
            db[im_id]['cand_id'] = np.arange(relationships_unique.shape[0], dtype=np.int32)

            # Add (human, object) negative pairs
            if self.neg_GT:
                obj_classes = db[im_id]['obj_gt_classes']
                sub_id = np.where(obj_classes==1)[0] # humans
                obj_id = np.where(obj_classes>=1)[0] # objects (included human)
                all_pairs = np.zeros((len(sub_id)*len(obj_id),2), dtype=np.int32)
                all_pairs[:,0] = np.repeat(sub_id, len(obj_id))
                all_pairs[:,1] = np.tile(obj_id, len(sub_id))
                is_pos = []
                for j in range(relationships_unique.shape[0]):
                    idx = np.where(np.logical_and((all_pairs[:,0]==relationships_unique[j,0]), (all_pairs[:,1]==relationships_unique[j,1])) >0)[0]
                    if len(idx)>0:
                        is_pos.append(idx[0])
                is_neg = np.setdiff1d(np.arange(all_pairs.shape[0]), is_pos)
                neg_pairs = all_pairs[is_neg,:]

                idx = np.where(neg_pairs[:,0]==neg_pairs[:,1])[0] # Discard candidates where subject==object box
                neg_pairs = np.delete(neg_pairs, idx, 0)

                gt_indicator = np.ones((neg_pairs.shape[0]), np.bool)
                cand_id_current = np.max(db[im_id]['cand_id']) + 1 if len(db[im_id]['cand_id'])>0 else 0

                db[im_id]['pair_ids'] = np.vstack((db[im_id]['pair_ids'], neg_pairs))
                db[im_id]['is_gt_pair'] = np.hstack((db[im_id]['is_gt_pair'], gt_indicator)) # it's not a gt pair, but it's made of gt boxes... 
                db[im_id]['cand_id'] = np.hstack((db[im_id]['cand_id'], cand_id_current + np.arange(neg_pairs.shape[0], dtype=np.int32)))


                # Labels the negative pairs

                # Some of these negative pairs intersect a gt: label them !!
                iou_pos = 0.5
                idx_pos_pair = np.where(np.sum(db[im_id]['labels_r'][:,1:],1)>=1)[0]
                neg_labels, neg_iou = self.build_label(neg_pairs, db[im_id]['pair_ids'][idx_pos_pair], \
                                                    db[im_id]['labels_r'][idx_pos_pair,:].toarray(), \
                                                    db[im_id]['boxes'], db[im_id]['obj_gt_classes'], iou_pos)

                db[im_id]['labels_r'] = lil_matrix(np.vstack((db[im_id]['labels_r'].toarray(), neg_labels)))



    def populate_candidates(self):
        """
        Get all candidate pairs from detections (do not filter by object scores at this stage)
        """
        #cand_boxes = json.load(open(self.cand_dir + '/' + 'detections_cocoa2014_results.json','rb'))
        cand_boxes = json.load(open(osp.join(self.cand_dir, 'bbox_cocoa_results.json'),'rb'))

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
                    obj_gt_class = gt_classes[id_max_ovl]
                    obj_labels_sr = gt_labels_sr[id_max_ovl,:].toarray()
                    obj_labels_ro = gt_labels_ro[id_max_ovl,:].toarray()


            # Append in database
            self.db[im_id]['boxes'] = np.vstack((self.db[im_id]['boxes'], np.array(list([x1,y1,x2,y2])) ))
            self.db[im_id]['obj_classes'] = np.hstack((self.db[im_id]['obj_classes'], np.array([obj_cat])))
            self.db[im_id]['obj_gt_classes'] = np.hstack((self.db[im_id]['obj_gt_classes'], np.array([obj_gt_class])))
            self.db[im_id]['obj_scores'] = np.hstack((self.db[im_id]['obj_scores'], np.array([score])))
            self.db[im_id]['is_gt'] = np.hstack((self.db[im_id]['is_gt'], np.zeros((1), dtype=np.bool)))
            self.db[im_id]['obj_id'] = np.hstack((self.db[im_id]['obj_id'], np.array([obj_id], dtype=np.int32)))
            self.db[im_id]['labels_sr'] = lil_matrix(np.vstack((self.db[im_id]['labels_sr'].toarray(), obj_labels_sr)))
            self.db[im_id]['labels_ro'] = lil_matrix(np.vstack((self.db[im_id]['labels_ro'].toarray(), obj_labels_ro)))


    def label_candidates(self):

        # Test : storing labels in scipy sparse matrix
        for im_id in self.db.keys():

            # All objects in image
            boxes = self.db[im_id]['boxes']
            obj_classes = self.db[im_id]['obj_classes']
            is_gt = self.db[im_id]['is_gt']

            idx_cand = np.where(is_gt==0)[0]
            idx_gt = np.where(is_gt==1)[0]

            if len(idx_cand)==0:
                continue

            assert np.max(idx_gt) < np.min(idx_cand), 'Warning db not in order'
            assert np.all(self.db[im_id]['is_gt_pair']==1), 'Warning some pair not GT'

            # Get the groundtruth annotations for this image
            is_gt_pair = self.db[im_id]['is_gt_pair']
            gt_pair_ids = self.db[im_id]['pair_ids']
            gt_pair_labels = self.db[im_id]['labels_r'].toarray()
            gt_cand_id = self.db[im_id]['cand_id']
            #pair_iou = self.db[im_id]['pair_iou']
            current_cand_id = np.max(gt_cand_id)+1 if len(gt_cand_id)>0 else 0

            # Form candidate pairs
            ids_subject = np.where(np.logical_and(obj_classes==1, is_gt==0))[0] # candidate humans
            ids_object = np.where(np.logical_and(obj_classes>=1, is_gt==0))[0] # all objects included human, excluding bg
            cand_pair_ids = np.zeros((len(ids_subject)*len(ids_object),2), dtype=np.int32)
            cand_pair_ids[:,0] = np.repeat(ids_subject, len(ids_object))
            cand_pair_ids[:,1] = np.tile(ids_object, len(ids_subject))

            idx = np.where(cand_pair_ids[:,0]==cand_pair_ids[:,1])[0] # Discard candidates where subject==object box
            cand_pair_ids = np.delete(cand_pair_ids, idx, 0)

            # Label subject-object relation
            idx_pos_pair = np.where(np.sum(gt_pair_labels[:,1:],1)>=1)[0]
            gt_pos_pair_ids = gt_pair_ids[idx_pos_pair,:]
            gt_pos_pair_labels = gt_pair_labels[idx_pos_pair,:]
            cand_pair_labels, cand_pair_iou = self.build_label(cand_pair_ids, gt_pos_pair_ids, gt_pos_pair_labels, boxes, obj_classes, self.iou_pos)

            # Merge candidates with GT
            self.db[im_id]['pair_ids'] = np.vstack((gt_pair_ids, cand_pair_ids))
            self.db[im_id]['labels_r'] = lil_matrix(np.vstack((gt_pair_labels, cand_pair_labels)))
            self.db[im_id]['is_gt_pair'] = np.hstack((is_gt_pair, np.zeros((cand_pair_ids.shape[0]),dtype=np.bool)))
            self.db[im_id]['cand_id'] = np.hstack((gt_cand_id, current_cand_id+np.arange(cand_pair_ids.shape[0], dtype=np.int32) ))
            #self.db[im_id]['pair_iou'] = np.vstack((pair_iou, cand_pair_iou))


    def create_splits(self, all_image_ids):
        """ Create splits : 1000 test, 1000 val, the rest for training """
        np.random.seed(0)

        # Do not use np.random.randint as there is replacement -> rather random choice or shuffle + select top first
        idx_test = np.random.choice(len(all_image_ids), size=1000, replace=False)
        image_ids_test = all_image_ids[idx_test]

        idx_trainval = np.setdiff1d(np.arange(len(all_image_ids)), idx_test)
        image_ids_trainval = all_image_ids[idx_trainval]

        idx_val = np.random.choice(len(image_ids_trainval), size=1000, replace=False)
        image_ids_val = image_ids_trainval[idx_val]

        idx_train = np.setdiff1d(np.arange(len(image_ids_trainval)), idx_val)
        image_ids_train = image_ids_trainval[idx_train]

        # We add a debug split : 100 images
        idx_debug = np.random.choice(len(all_image_ids), size=100, replace=False)
        image_ids_debug = all_image_ids[idx_debug]


        image_ids = {'all':all_image_ids, 'val':image_ids_val, 'train':image_ids_train, 'trainval':image_ids_trainval, 'test':image_ids_test, 'debug':image_ids_debug}

        for split in ['all','val','train','test','trainval','debug']:

            filename = osp.join(self.data_dir, split + '.ids')

            if not osp.exists(filename):
                np.savetxt(filename, image_ids[split], fmt='%d')


    def build_label(self, cand_pair_ids, gt_pair_ids, gt_pair_labels, boxes, obj_classes, iou_pos):

        cand_pair_labels = np.zeros((len(cand_pair_ids), len(self.predicates)))
        cand_pair_iou = np.zeros((len(cand_pair_ids),2))

        ids_subject = cand_pair_ids[:,0]
        ids_object = cand_pair_ids[:,1]

        # Scan the groundtruth relationships for this image and mark as positives candidates overlapping
        for j in range(gt_pair_ids.shape[0]):
            gt_sub = gt_pair_ids[j,0]
            gt_obj = gt_pair_ids[j,1]
            obj_cat = obj_classes[gt_obj]
            subject_box = boxes[gt_sub,:]
            object_box = boxes[gt_obj,:]

            # Filter candidates by category
            idx = np.where(obj_classes[ids_object]==obj_cat)[0]
            if len(idx)==0:
                continue

            # Overlap with candidates
            ovl_subject = get_overlap(boxes[ids_subject,:], subject_box)
            ovl_object = get_overlap(boxes[ids_object[idx],:], object_box)

            # Fill overlap for both positives and negatives
            cand_pair_iou[:,0] = np.maximum(cand_pair_iou[:,0], ovl_subject)
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


    def print_occurrences(self):
        """ Compute occurrences of visualphrases over trainval/test """
        images = {  'trainval':self.get_image_ids('trainval'),\
                    'test':self.get_image_ids('test')} 

        occ_visualphrases = {   'trainval':{triplet:0 for triplet in self.visualphrases.words()}, \
                                'test':{triplet:0 for triplet in self.visualphrases.words()}}

        for annot in self.annotations:

            im_id = annot['image_id']

            if im_id in images['trainval']:
                split = 'trainval'
            elif im_id in images['test']:
                split = 'test'
            else:
                print('Pb')

            subject_id_coco = annot['subject_id']
            object_id_coco = annot['object_id']

            if object_id_coco==-1:
                continue

            visual_actions = annot['visual_actions']

            # Get annotations in COCO
            subject_anns = self.COCO.loadAnns(subject_id_coco)[0]
            subjectname  = self.COCO.cats[subject_anns['category_id']]['name']
            object_anns  = self.COCO.loadAnns(object_id_coco)[0]
            objectname   = self.COCO.cats[object_anns['category_id']]['name']

            for action_id in visual_actions:
                predicate = self.id_to_action[action_id]
                rel_cat = self.predicates.word2idx[predicate]
                triplet = '-'.join([subjectname, predicate, objectname])
                occ_visualphrases[split][triplet] +=1

        # Print in csv
        csv_file = osp.join(self.data_dir, "occurrences.csv")
        with open(csv_file, mode='w') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',')
            csv_writer.writerow(['triplet','occ_trainval','occ_test'])
            for triplet in self.visualphrases.words():
                csv_writer.writerow([triplet, occ_visualphrases['trainval'][triplet], occ_visualphrases['test'][triplet]])

        # And save pickle
        pickle.dump(occ_visualphrases, open(osp.join(self.data_dir, "occurrences.pkl"),'wb'))


        # Print the zero-shot triplets in a separate file
        zeroshot_triplets = []
        for triplet in self.visualphrases.words():
            if occ_visualphrases['trainval'][triplet]==0 and occ_visualphrases['test'][triplet]>0:
                zeroshot_triplets.append(triplet)

        # And save pickle
        pickle.dump(zeroshot_triplets, open(osp.join(self.data_dir, "zeroshot_triplets.pkl"),'wb'))


    def get_occurrences_precomp(self, split):
        """ 
        Get number of triplets annotated in split 
        """
        filename = osp.join(self.data_dir, 'occurrences.csv')
        occurrences = {}
        with open(filename) as f:
            reader = csv.DictReader(f)
            for line in reader:
                
                triplet = line['triplet']

                if split=='all':
                    occurrences[triplet] = int(line['occ_trainval']) + int(line['occ_test'])
                else:
                    occ_split = line['occ_' + split]
                    occurrences[triplet] = int(occ_split)

        return occurrences


    def get_triplets(self):
        """ Also get the occurrence of each triplet """
        triplets = []
        occ_triplets = {}

        for annot in self.annotations:

            im_id = annot['image_id']

            if im_id not in self.image_ids:
                continue

            subject_id_coco = annot['subject_id']
            object_id_coco = annot['object_id']

            if object_id_coco==-1:
                continue

            visual_actions = annot['visual_actions']

            # Get annotations in COCO
            subject_anns = self.COCO.loadAnns(subject_id_coco)[0]
            subjectname  = self.COCO.cats[subject_anns['category_id']]['name']
            object_anns  = self.COCO.loadAnns(object_id_coco)[0]
            objectname   = self.COCO.cats[object_anns['category_id']]['name']

            for action_id in visual_actions:
                predicate = self.id_to_action[action_id]
                rel_cat = self.predicates.word2idx[predicate]
                triplet = '-'.join([subjectname, predicate, objectname])
                if triplet not in triplets:
                    triplets.append(triplet)
                    occ_triplets[triplet] = 0
                occ_triplets[triplet] +=1

        return triplets, occ_triplets 

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
        print('Loading the GT annotations...')
        count = 0
        for annot in self.annotations:

            if count%100==0:
                print('Load GT {}/{}'.format(count, len(self.annotations)))

            im_id = annot['image_id']

            if im_id not in self.image_ids:
                continue

            subject_id_coco = annot['subject_id']
            object_id_coco = annot['object_id']

            if object_id_coco==-1:
                continue

            visual_actions = annot['visual_actions']


            # Get annotations in COCO
            subject_anns = self.COCO.loadAnns(subject_id_coco)[0]
            subjectname  = self.COCO.cats[subject_anns['category_id']]['name']
            object_anns  = self.COCO.loadAnns(object_id_coco)[0]
            objectname   = self.COCO.cats[object_anns['category_id']]['name']

            # Convert form x1, y1, w, h to x1, y1, x2, y2
            x1 = subject_anns['bbox'][0]
            y1 = subject_anns['bbox'][1]
            x2 = x1 + np.maximum(0., subject_anns['bbox'][2] - 1.)
            y2 = y1 + np.maximum(0., subject_anns['bbox'][3] - 1.)
            subject_box = np.array([x1,y1,x2,y2])
            
            x1 = object_anns['bbox'][0]
            y1 = object_anns['bbox'][1]
            x2 = x1 + np.maximum(0., object_anns['bbox'][2] - 1.)
            y2 = y1 + np.maximum(0., object_anns['bbox'][3] - 1.)
            object_box = np.array([x1,y1,x2,y2])


            for action_id in visual_actions:
                predicate = self.id_to_action[action_id]
                rel_cat = self.predicates.word2idx[predicate]
                triplet = '-'.join([subjectname, predicate, objectname])
                if triplet in triplets:
                    gt[triplet][im_id] = np.vstack((gt[triplet][im_id], np.hstack((subject_box, object_box))))
                    npos[triplet] += 1

            count +=1


        """ Match the detections """

        print('Matching detections with GT...')

        for t in range(len(triplets)):

            if t%100==0:
                print('Done triplet {}/{}'.format(t,len(triplets)))

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

    def eval_speed(self, dets_triplet, gt_triplet, npos_triplet, min_overlap=0.5):

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
        rec = tp / float(npos_triplet)
        # ground truth
        prec = tp / (tp + fp)
        #prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = self.voc_ap(rec, prec, use_07_metric=False)
        if len(rec)>0:
            recall = max(rec) 
        else:
            recall = 0

        return ap, recall



    def get_gt(self, triplets):

        gt = {}
        npos = {} # number of positives for each triplet
        for triplet in triplets:
            gt[triplet] = {im_id:np.empty((0,8)) for im_id in self.image_ids}
            npos[triplet] = 0
    
        """ Get the GT annotations for each triplet """
        print('Loading the GT annotations...')
        count = 0
        for annot in self.annotations:

            if count%100==0:
                print('Load GT {}/{}'.format(count, len(self.annotations)))

            im_id = annot['image_id']

            if im_id not in self.image_ids:
                continue

            subject_id_coco = annot['subject_id']
            object_id_coco = annot['object_id']

            if object_id_coco==-1:
                continue

            visual_actions = annot['visual_actions']


            # Get annotations in COCO
            subject_anns = self.COCO.loadAnns(subject_id_coco)[0]
            subjectname  = self.COCO.cats[subject_anns['category_id']]['name']
            object_anns  = self.COCO.loadAnns(object_id_coco)[0]
            objectname   = self.COCO.cats[object_anns['category_id']]['name']

            # Convert form x1, y1, w, h to x1, y1, x2, y2
            x1 = subject_anns['bbox'][0]
            y1 = subject_anns['bbox'][1]
            x2 = x1 + np.maximum(0., subject_anns['bbox'][2] - 1.)
            y2 = y1 + np.maximum(0., subject_anns['bbox'][3] - 1.)
            subject_box = np.array([x1,y1,x2,y2])
            
            x1 = object_anns['bbox'][0]
            y1 = object_anns['bbox'][1]
            x2 = x1 + np.maximum(0., object_anns['bbox'][2] - 1.)
            y2 = y1 + np.maximum(0., object_anns['bbox'][3] - 1.)
            object_box = np.array([x1,y1,x2,y2])


            for action_id in visual_actions:
                predicate = self.id_to_action[action_id]
                rel_cat = self.predicates.word2idx[predicate]
                triplet = '-'.join([subjectname, predicate, objectname])
                if triplet in triplets:
                    gt[triplet][im_id] = np.vstack((gt[triplet][im_id], np.hstack((subject_box, object_box))))
                    npos[triplet] += 1

            count +=1

        return gt, npos


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


    @staticmethod
    def clip_xyxy_to_image(x1, y1, x2, y2, height, width):
        x1 = np.minimum(width - 1., np.maximum(0., x1))
        y1 = np.minimum(height - 1., np.maximum(0., y1))
        x2 = np.minimum(width - 1., np.maximum(0., x2))
        y2 = np.minimum(height - 1., np.maximum(0., y2))
        return x1, y1, x2, y2


    def get_triplets_subset(self, subset):

        if subset=='all':
            triplets = self.visualphrases.words()

        elif subset=='unseen':
            triplets = pickle.load(open(osp.join(self.data_dir, 'unseen_triplets.pkl'),'rb'))

        elif subset=='outofvocab':
            triplets = pickle.load(open(osp.join(self.data_dir, 'out_of_vocabulary_triplets.pkl'), 'rb'))

        elif subset=='debug':
            triplets = ['person-ride-horse', 'person-hold-sports ball','person-ride-skateboard','person-sit-chair','person-pet-dog']

        return triplets



