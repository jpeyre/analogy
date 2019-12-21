import numpy as np
import torch
import torch.utils.data as data

class TestSampler(data.Dataset):
    def __init__(self, dset, sampler_name=None, use_image=False, use_precompappearance=False, use_precompobjectscore=False):

        self.dset = dset # dataset class
        self.sampler_name = sampler_name
        self.use_image = use_image
        self.use_precompappearance = use_precompappearance
        self.use_precompobjectscore = use_precompobjectscore


    def __getitem__(self, index):

        im_id = self.dset.candidates[index,0]
        cand_idx = self.dset.candidates[index,1]

        """ Init input matrices """
        cand_info = np.empty((1,2), dtype=np.int32)
        image = []
        precompappearance = []
        precompobjectscore = []

        """ Fill input matrices """
        cand_info[:,0] = im_id
        cand_info[:,1] = cand_idx
        cand_info      = cand_info.astype(np.int32)
        pair_objects   = self.dset.load_pair_objects(im_id, cand_idx)
        labels_s       = self.dset.get_labels_subjects(im_id, cand_idx)
        labels_o       = self.dset.get_labels_objects(im_id, cand_idx)
        labels_r       = self.dset.get_labels_predicates(im_id, cand_idx)
        labels_sr      = self.dset.get_labels_subjectpredicates(im_id, cand_idx)
        labels_ro      = self.dset.get_labels_objectpredicates(im_id, cand_idx)
        labels_sro     = self.dset.get_labels_visualphrases(im_id, cand_idx)


        if self.use_precompappearance:
            precompappearance = self.dset.load_appearance(im_id, cand_idx)

        if self.use_precompobjectscore:
            precompobjectscore = self.dset.load_objectscores(im_id, cand_idx)

        if self.use_image:
            image = self.dset.load_context_image(im_id)

        return cand_info, pair_objects, image, precompappearance, precompobjectscore, labels_s, labels_o, labels_r, labels_sr, labels_ro, labels_sro



    def __len__(self):
        return self.dset.candidates.shape[0]


    def collate_fn(self, data):
        """Creates mini-batch tensors from the list of tuples ().
        """
        cand_info, pair_objects, image, precompappearance, precompobjectscore, labels_s, labels_o, labels_r, labels_sr, labels_ro, labels_sro = zip(*data)


        output = {}
        num_batches = len(cand_info)
        num_instances = len(cand_info[0])

        output['cand_info']    = torch.from_numpy(np.concatenate(cand_info, axis=0))
        output['pair_objects'] = torch.from_numpy(np.concatenate(pair_objects, axis=0)).float()
        output['labels_s']     = torch.from_numpy(np.concatenate(labels_s, axis=0)).long()
        output['labels_o']     = torch.from_numpy(np.concatenate(labels_o, axis=0)).long()
        output['labels_r']     = torch.from_numpy(np.concatenate(labels_r, axis=0)).long()
        output['labels_sr']    = torch.from_numpy(np.concatenate(labels_sr, axis=0)).long()
        output['labels_ro']    = torch.from_numpy(np.concatenate(labels_ro, axis=0)).long()
        output['labels_sro']   = torch.from_numpy(np.concatenate(labels_sro, axis=0)).long()

        if self.use_precompappearance:
            output['precompappearance'] = torch.from_numpy(np.concatenate(precompappearance, axis=0)).float()

        if self.use_precompobjectscore:
            output['precompobjectscore'] = torch.from_numpy(np.concatenate(precompobjectscore, axis=0)).float()

        if self.use_image:
            output['image'] = torch.from_numpy(np.concatenate(image, axis=0)).float()

        return output



class TrainSampler(data.Dataset):
    def __init__(self, dset, sampler_name, num_negatives=3, use_image=False, use_precompappearance=False, use_precompobjectscore=False):

        self.dset = dset # data class
        self.sampler_name = sampler_name
        self.num_negatives = num_negatives
        self.use_image = use_image
        self.use_precompappearance = use_precompappearance
        self.use_precompobjectscore = use_precompobjectscore
        self.num_pos = 1
        self.num_neg = self.num_negatives


    def __getitem__(self, index):


        num_pos = self.num_pos
        num_neg = self.num_neg
        N = num_pos + num_neg 

        """ Intialize input matrices to empty """

        inputs = {}
        inputs['cand_info']             = np.empty((N,2))
        inputs['labels_s']              = np.empty((N, len(self.dset.classes)))
        inputs['labels_o']              = np.empty((N, len(self.dset.classes)))
        inputs['labels_r']              = np.empty((N, len(self.dset.predicates)))
        inputs['labels_sr']             = np.empty((N, len(self.dset.subjectpredicates)))
        inputs['labels_ro']             = np.empty((N, len(self.dset.objectpredicates)))
        inputs['labels_sro']            = np.empty((N, len(self.dset.visualphrases)))
        inputs['pair_objects']          = np.empty((N,2,6)) # [x1,y1,x2,y2,obj_cat,conf_score]
        inputs['image']                 = []
        inputs['precompappearance']     = []
        inputs['precompobjectscore']    = []
       
        if self.use_precompappearance:
            inputs['precompappearance'] = np.empty((N,2,self.dset.d_appearance))
        if self.use_precompobjectscore:
            inputs['precompobjectscore'] = np.empty((N,2,len(self.dset.classes))) 
        if self.use_image:
            inputs['image'] = np.empty((N,3,224,224))
       
 
        """ Sample training batch """
        id_sample = 0

        # Get the positive instance
        im_id, cand_id, sub_cat, obj_cat = self.dset.cand_positives[index,:]

        cand_info = np.empty((N,2))
        cand_info[id_sample,0] = im_id
        cand_info[id_sample,1] = cand_id
        id_sample +=1


        # Sample negatives in the batch (can use different strategies specified by sampler_name)
        if self.sampler_name == 'priority_object':
            """
            This strategy: sample negatives involving the same object category in other images
            """

            id_pos_object = self.dset.get_pair_ids(im_id, idx=cand_id)[:,1]

            # If possible sample negatives from this image, with this object and other human
            num_neg_sampled = 0

            # Sample additional negatives involving the same object category in other images
            if num_neg_sampled < num_neg:
                idx_match_object = self.dset.idx_match_object_candneg[obj_cat] 

                if len(idx_match_object)>0:
                    idx_neg,_ = self.sample_random_negatives(idx_match_object, \
                                                             min(len(idx_match_object),num_neg-num_neg_sampled))
                    cand_info[id_sample:id_sample+len(idx_neg),:] = self.dset.cand_negatives[idx_neg,:2]
                    num_neg_sampled += len(idx_neg)
                    id_sample += len(idx_neg)

            # Sample additional negatives randomly from other images (not necessarily with same object category)
            if num_neg_sampled < num_neg:
                idx_neg,_ = self.sample_random_negatives(np.arange(len(self.dset.cand_negatives)), \
                                                         num_neg-num_neg_sampled)
                cand_info[id_sample:id_sample+len(idx_neg),:] = self.dset.cand_negatives[idx_neg,:2]
                num_neg_sampled += len(idx_neg)
                id_sample += len(idx_neg)


        # Negatives are sampled according to strategy
        elif self.sampler_name == 'priority_subjectobject':
            """
            This strategy: sample negatives involving the same object category in other images. Also sample from same subject category
            """

            num_neg_sampled = 0

            # Sample additional negatives involving the same object category in other images
            if num_neg_sampled < num_neg:

                idx_match_object = self.dset.idx_match_object_candneg[obj_cat]
                idx_match_subject = self.dset.idx_match_subject_candneg[sub_cat]

                # Try sample 1 involving same object, different subject
                idx_match_subjectobject = np.intersect1d(idx_match_object, idx_match_subject) 
                if len(idx_match_subjectobject)>0:
                    idx_neg,_ = self.sample_random_negatives(idx_match_subjectobject, \
                                                             min(len(idx_match_subjectobject), num_neg-num_neg_sampled))
                    cand_info[id_sample:id_sample+len(idx_neg),:] = self.dset.cand_negatives[idx_neg,:2]
                    num_neg_sampled += len(idx_neg)
                    id_sample += len(idx_neg)


            # Sample additional negatives randomly from other images (not necessarily with same object category)
            if num_neg_sampled < num_neg:
                idx_neg,_ = self.sample_random_negatives(np.arange(len(self.dset.cand_negatives)), \
                                                         num_neg-num_neg_sampled)
                cand_info[id_sample:id_sample+len(idx_neg),:] = self.dset.cand_negatives[idx_neg,:2]
                num_neg_sampled += len(idx_neg)
                id_sample += len(idx_neg)


        elif self.sampler_name == 'random':
            """
            Sample negatives totally at random
            """ 
            num_neg_sampled = 0

            # Sample additional negatives randomly from other images
            if num_neg_sampled < num_neg:
                idx_neg = np.random.choice(np.arange(len(self.dset.cand_negatives)), \
                                           num_neg-num_neg_sampled , replace=False) # sample in all candidates
                cand_info[id_sample:id_sample+len(idx_neg),:] = self.dset.cand_negatives[idx_neg,:2]
                num_neg_sampled += len(idx_neg)


        assert id_sample-1==num_neg, 'warning: you did not sample the right number of negatives'


        """ Fill the batch """

        cand_info = cand_info.astype(np.int32)

        for j in range(cand_info.shape[0]):
            
            im_id, cand_id              = cand_info[j,:]
            pair_objects_current        = self.dset.load_pair_objects(im_id, cand_id)
            inputs['cand_info'][j]      = cand_info[j,:]
            inputs['pair_objects'][j]   = pair_objects_current
            inputs['labels_s'][j]       = self.dset.get_labels_subjects(im_id, cand_id)
            inputs['labels_o'][j]       = self.dset.get_labels_objects(im_id, cand_id)
            inputs['labels_r'][j]       = self.dset.get_labels_predicates(im_id, cand_id)
            inputs['labels_sr'][j]      = self.dset.get_labels_subjectpredicates(im_id, cand_id)
            inputs['labels_ro'][j]      = self.dset.get_labels_objectpredicates(im_id, cand_id)
            inputs['labels_sro'][j]     = self.dset.get_labels_visualphrases(im_id, cand_id)

            if self.use_precompappearance:
                inputs['precompappearance'][j] = self.dset.load_appearance(im_id, cand_id)

            if self.use_precompobjectscore:
                inputs['precompobjectscore'][j] = self.dset.load_objectscores(im_id, cand_id)

            if self.use_image:
                inputs['image'][j] = self.dset.load_context_image(im_id)



        return inputs['cand_info'], inputs['pair_objects'], inputs['image'], \
                inputs['precompappearance'], inputs['precompobjectscore'], \
                inputs['labels_s'], inputs['labels_o'], inputs['labels_r'], \
                inputs['labels_sr'], inputs['labels_ro'], inputs['labels_sro']
 


    def __len__(self):
        return self.dset.cand_positives.shape[0]

    def collate_fn(self, data):
        """Creates mini-batch tensors from the list of tuples ().
        """
        cand_info, pair_objects, image, precompappearance, precompobjectscore, labels_s, labels_o, labels_r, labels_sr, labels_ro, labels_sro = zip(*data)
        
        output = {}
        num_batches = len(cand_info)
        num_instances = len(cand_info[0])

        output['cand_info']    = torch.from_numpy(np.concatenate(cand_info, axis=0))
        output['pair_objects'] = torch.from_numpy(np.concatenate(pair_objects, axis=0)).float()
        output['labels_s']     = torch.from_numpy(np.concatenate(labels_s, axis=0)).long()
        output['labels_o']     = torch.from_numpy(np.concatenate(labels_o, axis=0)).long()
        output['labels_r']     = torch.from_numpy(np.concatenate(labels_r, axis=0)).long()
        output['labels_sr']    = torch.from_numpy(np.concatenate(labels_sr, axis=0)).long()
        output['labels_ro']    = torch.from_numpy(np.concatenate(labels_ro, axis=0)).long()
        output['labels_sro']   = torch.from_numpy(np.concatenate(labels_sro, axis=0)).long()

        if self.use_precompappearance:
            output['precompappearance'] = torch.from_numpy(np.concatenate(precompappearance, axis=0)).float()
        
        if self.use_precompobjectscore:
            output['precompobjectscore'] = torch.from_numpy(np.concatenate(precompobjectscore, axis=0)).float()

        if self.use_image:
            output['image'] = torch.from_numpy(np.concatenate(image, axis=0)).float()

        return output


    def sample_random_negatives(self, idx, num_to_sample):
        idx_sample = np.random.randint(0, len(idx), size=num_to_sample)
        idx_values = idx[idx_sample]
        return idx_values, idx_sample



 
