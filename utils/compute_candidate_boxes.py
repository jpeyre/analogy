"""
Get candidate object boxes (both GT and candidates) to provide to detectron to compute appearance/object scores
The way we have implemented this :
1. Run the object detector to get candidate object (we used Detectron)
2. Create database object merging candidate detections and groundtruth
3. Forward the bounding boxes into the object detector to get appearance features for both detections and GT (that we use at training)
This script allows you to build the database object and create a file "proposals.pkl" that you can use as candidate proposals
"""

import __init__
import numpy as np
import cPickle as pickle
import os.path as osp


DATA_PATH = '/sequoia/data2/jpeyre/iccv19_final/datasets'
data_name = 'hico' 

if data_name=='hico':
    from datasets.hico_api import Hico as Dataset
    splits = ['trainval','train','val','test']

elif data_name=='hicoforcocoa':
    from datasets.hico_api import Hico as Dataset
    splits = ['trainval','test']

elif data_name=='cocoa':
    from datasets.cocoa_api import Cocoa as Dataset
    splits = ['all']


data_path  = osp.join(DATA_PATH, data_name)
image_path = osp.join(data_path, 'images')
cand_dir   = osp.join(data_path, 'detections')


proposals = {}
for split in splits:

    dataset = Dataset(data_path, image_path, split, cand_dir=cand_dir,\
                 thresh_file='', use_gt=False, add_gt=True, train_mode=False, jittering=False, store_ram=[])

    for im_id in dataset.image_ids:

        cand_boxes       = dataset.get_boxes(im_id)
        obj_id           = dataset.get_obj_id(im_id)
        proposals[im_id] = np.hstack((obj_id[:,None], cand_boxes))

        assert len(np.unique(obj_id))==len(obj_id), 'Careful duplicate obj_id'

    pickle.dump(proposals, open(osp.join(cand_dir, split + '_proposals.pkl'),'wb'))



