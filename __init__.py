import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

# Add to path coco dir
coco_dir = './data/coco'

# Add pycocotools to PYTHONPATH
add_path('../')
add_path(coco_dir)
coco_path = osp.join(coco_dir, 'PythonAPI')
add_path(coco_path)

