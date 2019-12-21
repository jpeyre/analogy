"""
Many functions of this utils come from Mask R-CNN directory, we thank the authors for publicly releasing their code. 
Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""
from __future__ import division
import sys
import os
import math
import random
import numpy as np
import scipy.misc
import skimage.color
import skimage.io
from skimage.transform import resize
from scipy.ndimage import gaussian_filter
import pdb
import torch
import cPickle as pickle
import cv2



############################################################
#  Transforms
############################################################

def box_integer(box):
    """
    Input is Nx4 array of box coordinates in float 
    Output is Nx4 array of box coordinates in int
    Coordinates are in order (xmin,ymin,xmax,ymax)
    """
    box[:,:2] = np.floor(box[:,:2])
    box[:,2:] = np.ceil(box[:,2:])
    return box


def flip_horizontal(box, width, height):
    """
    Flip box horinzontally
    """
    x1,y1,x2,y2 = box[:,0], box[:,1], box[:,2], box[:,3]

    x1_flip = width-x2
    x2_flip = width-x1

    flip_box = np.stack((x1_flip, y1, x2_flip, y2),1)

    return flip_box

def jitter_boxes2(box, width, height):

    shift_jitter = 16
    scale_jitter = 0.25
    #iSz = np.sqrt(width*height)
    #objSz = math.ceil(iSz*128/224)
    #wSz = iSz + 32

    x1,y1,x2,y2 = box[:,0], box[:,1], box[:,2], box[:,3]
    w = x2-x1+1
    h = y2-y1+1
    xc, yc = x1+w/2, y1+h/2
    #maxDim = np.maximum(w,h)
    #scale = np.log2(maxDim/objSz)
    #s = scale + np.random.uniform(-scale_jitter, scale_jitter)
    s = np.random.uniform(-scale_jitter, scale_jitter)
    xc = xc + np.random.uniform(-shift_jitter, shift_jitter)*pow(2,s)
    yc = yc + np.random.uniform(-shift_jitter, shift_jitter)*pow(2,s)
    w, h = w*pow(2,s), h*pow(2,s)
    
    x1_new = xc-w/2
    y1_new = yc-h/2
    x2_new = xc+w/2
    y2_new = yc+h/2

    # Clip to image
    x1_new,y1_new,x2_new,y2_new = clip_xyxy_to_image(x1_new, y1_new, x2_new, y2_new, height, width)

    # Case where no transformation : x2-x1<1 or y2-y2<1
    if x2_new-x1_new < 1:
        x2_new = x2
        x1_new = x1

    if y2_new-y1_new < 1:
        y2_new = y2
        y1_new = y1

    jittered_box = np.stack((x1_new,y1_new,x2_new,y2_new),1)

    return jittered_box


def jitter_boxes(box, width, height):
    """
    Jitter box to simulate proposals: apply random small translation/rescaling
    Small: to keep IoU=0.5 with non jittered box -> could try to compute it, or just experimentally 
    By hand: imagine your object is square (nxn): if you shift box by unit 1 in same direction in both x and y (worst case), \
    then IoU is given by (n-1)*(n-1)/((n-1)*(n-1)+4*(n-1)+2). Setting n=5 provide worst IoU<0.5, setting n=6 gives IoU>0.5
    So we accept deformation 
    """

    x1,y1,x2,y2 = box[:,0], box[:,1], box[:,2], box[:,3]

    # Small translation (x,y)
    xt_min = -np.maximum((x2-x1)/5,1)
    xt_max = np.maximum((x2-x1)/5,1)
    yt_min = -np.maximum((y2-y1)/5,1)
    yt_max = np.maximum(1,(y2-y1)/5)
    x_trans = (xt_max-xt_min)*np.random.random(size=box.shape[0]) + xt_min
    y_trans = (yt_max-yt_min)*np.random.random(size=box.shape[0]) + yt_min

    # Transform
    x1_new = x1 + x_trans
    x2_new = x2 + x_trans
    y1_new = y1 + y_trans
    y2_new = y2 + y_trans 

    # Apply small rescaling: keep aspect ratio but scale it
    scale_factor = np.random.uniform(pow(2,-1/4), pow(2,1/4), size=box.shape[0]) # value taken from "Learning to segment object candidates"
    center_x = (x1_new+x2_new)/2
    center_y = (y1_new+y2_new)/2
    w_box = (x2_new-x1_new+1)
    h_box = (y2_new-y1_new+1)
    w_box_scale = w_box*scale_factor
    h_box_scale = h_box*scale_factor
    x1_new = center_x - w_box_scale/2
    x2_new = center_x + w_box_scale/2
    y1_new = center_y - h_box_scale/2
    y2_new = center_y + h_box_scale/2  

    # Clip to image
    x1_new,y1_new,x2_new,y2_new = clip_xyxy_to_image(x1_new, y1_new, x2_new, y2_new, height, width)

    # Case where no transformation : x2-x1<1 or y2-y2<1
    if x2_new-x1_new < 1:
        x2_new = x2
        x1_new = x1

    if y2_new-y1_new < 1:
        y2_new = y2
        y1_new = y1

    jittered_box = np.stack((x1_new,y1_new,x2_new,y2_new),1)

    return jittered_box


def filter_small_boxes(boxes, min_size):
    """Keep boxes with width and height both greater than min_size."""
    w = boxes[:, 2] - boxes[:, 0] + 1
    h = boxes[:, 3] - boxes[:, 1] + 1
    keep = np.where((w >= min_size) & (h >= min_size))[0]
    return keep


def clip_xyxy_to_image(x1, y1, x2, y2, height, width):
    """Clip coordinates to an image with the given height and width."""
    x1 = np.minimum(width - 1., np.maximum(0., x1))
    y1 = np.minimum(height - 1., np.maximum(0., y1))
    x2 = np.minimum(width - 1., np.maximum(0., x2))
    y2 = np.minimum(height - 1., np.maximum(0., y2))
    return x1, y1, x2, y2


def get_union_boxes(pair_boxes):
    """
    Input: array (N,2,4)
    """
    xu = np.minimum(pair_boxes[:,0,0], pair_boxes[:,1,0])
    yu = np.minimum(pair_boxes[:,0,1], pair_boxes[:,1,1])
    xu_end = np.maximum(pair_boxes[:,0,2], pair_boxes[:,1,2])  
    yu_end = np.maximum(pair_boxes[:,0,3], pair_boxes[:,1,3])
    union_boxes = np.stack((xu, yu, xu_end, yu_end), axis=1)
    return union_boxes


def box_to_mask(boxes, image_size):
    """
    Create binary masks from boxes : extent is all image
    Input: boxes: array (N,4)
            image_size : [W,H]
    Output: masks: array(N,H,W)
    """
    width, height = image_size
    N = boxes.shape[0]
    masks = np.zeros((height,width,N))
    for j in range(N):
        x1,y1,x2,y2 = boxes[j,:]
        masks[:,:,j][y1:y2+1,:][:,x1:x2+1].fill(1)

    return masks


 
def multilabel_transform(relationships, num_labels):
    """
    Take as input an array of dim Nx3 = [sub_id, obj_id, rel_cat]
    Return array of unique [sub_id, obj_id, labels] where labels is of length num_predicates
    """

    relationships = relationships.astype(int)
    unique_ids, unique_inverse = np.unique(relationships[:,:2], return_inverse=True, axis=0)
    labels = np.zeros((unique_ids.shape[0], num_labels))
    labels[unique_inverse, relationships[:,2]] = 1
    unique_relationships = np.hstack((unique_ids, labels))    

    return unique_relationships


def sample_negatives(objects, relationships, proportion=0.75):

    num_pos = relationships.shape[0]
    num_neg = int(proportion/(1-proportion)*num_pos)
    relationships_neg = np.zeros((num_neg, relationships.shape[1]))

    sub_id = np.where(objects[:,4]==1)[0] # humans
    obj_id = np.where(objects[:,4]!=1)[0] # not human
    all_pairs = np.zeros((len(sub_id)*len(obj_id),2))
    all_pairs[:,0] = np.repeat(sub_id, len(obj_id))
    all_pairs[:,1] = np.tile(obj_id, len(sub_id))
    is_pos = []
    for j in range(relationships.shape[0]):
        idx = np.where(np.logical_and((all_pairs[:,0]==relationships[j,0]), (all_pairs[:,1]==relationships[j,1])) >0)[0]
        if len(idx)>0:
            is_pos.append(idx[0])
    is_neg = np.setdiff1d(np.arange(all_pairs.shape[0]), is_pos)
    
    if len(is_neg)==0:
        return relationships
    elif len(is_neg) >= num_neg:
        idx = np.random.choice(is_neg, size=num_neg, replace=False)
    else:
        idx = np.random.choice(is_neg, size=num_neg, replace=True)

    relationships_neg[:,:2] = all_pairs[idx,:]
    relationships = np.vstack((relationships, relationships_neg))

    return relationships


############################################################
#  Bounding Boxes
############################################################

def get_overlap(boxes, ref_box):
    ixmin = np.maximum(boxes[:, 0], ref_box[0])
    iymin = np.maximum(boxes[:, 1], ref_box[1])
    ixmax = np.minimum(boxes[:, 2], ref_box[2])
    iymax = np.minimum(boxes[:, 3], ref_box[3])
    iw = np.maximum(ixmax - ixmin + 1., 0.)
    ih = np.maximum(iymax - iymin + 1., 0.)
    inters = iw * ih

    # union
    uni = ((ref_box[2] - ref_box[0] + 1.) * (ref_box[3] - ref_box[1] + 1.) +\
        (boxes[:, 2] - boxes[:, 0] + 1.) *(boxes[:, 3] - boxes[:, 1] + 1.) - inters)

    overlaps = inters / uni
    return overlaps



def compute_iou(box, boxes, box_area, boxes_area):
    """Calculates IoU of the given box with the array of the given boxes.
    box: 1D vector [y1, x1, y2, x2]
    boxes: [boxes_count, (y1, x1, y2, x2)]
    box_area: float. the area of 'box'
    boxes_area: array of length boxes_count.

    Note: the areas are passed in rather than calculated here for
          efficency. Calculate once in the caller to avoid duplicate work.
    """
    # Calculate intersection areas
    y1 = np.maximum(box[0], boxes[:, 0])
    y2 = np.minimum(box[2], boxes[:, 2])
    x1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    union = box_area + boxes_area[:] - intersection[:]
    iou = intersection / union
    return iou


def compute_overlaps(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (y1, x1, y2, x2)].

    For better performance, pass the largest set first and the smaller second.
    """
    # Areas of anchors and GT boxes
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Compute overlaps to generate matrix [boxes1 count, boxes2 count]
    # Each cell contains the IoU value.
    overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0]))
    for i in range(overlaps.shape[1]):
        box2 = boxes2[i]
        overlaps[:, i] = compute_iou(box2, boxes1, area2[i], area1)
    return overlaps


def non_max_suppression(boxes, scores, threshold):
    """Performs non-maximum supression and returns indicies of kept boxes.
    boxes: [N, (y1, x1, y2, x2)]. Notice that (y2, x2) lays outside the box.
    scores: 1-D array of box scores.
    threshold: Float. IoU threshold to use for filtering.
    """
    assert boxes.shape[0] > 0
    if boxes.dtype.kind != "f":
        boxes = boxes.astype(np.float32)

    # Compute box areas
    y1 = boxes[:, 0]
    x1 = boxes[:, 1]
    y2 = boxes[:, 2]
    x2 = boxes[:, 3]
    area = (y2 - y1) * (x2 - x1)

    # Get indicies of boxes sorted by scores (highest first)
    ixs = scores.argsort()[::-1]

    pick = []
    while len(ixs) > 0:
        # Pick top box and add its index to the list
        i = ixs[0]
        pick.append(i)
        # Compute IoU of the picked box with the rest
        iou = compute_iou(boxes[i], boxes[ixs[1:]], area[i], area[ixs[1:]])
        # Identify boxes with IoU over the threshold. This
        # returns indicies into ixs[1:], so add 1 to get
        # indicies into ixs.
        remove_ixs = np.where(iou > threshold)[0] + 1
        # Remove indicies of the picked and overlapped boxes.
        ixs = np.delete(ixs, remove_ixs)
        ixs = np.delete(ixs, 0)
    return np.array(pick, dtype=np.int32)


def apply_box_deltas(boxes, deltas):
    """Applies the given deltas to the given boxes.
    boxes: [N, (y1, x1, y2, x2)]. Note that (y2, x2) is outside the box.
    deltas: [N, (dy, dx, log(dh), log(dw))]
    """
    boxes = boxes.astype(np.float32)
    # Convert to y, x, h, w
    height = boxes[:, 2] - boxes[:, 0]
    width = boxes[:, 3] - boxes[:, 1]
    center_y = boxes[:, 0] + 0.5 * height
    center_x = boxes[:, 1] + 0.5 * width
    # Apply deltas
    center_y += deltas[:, 0] * height
    center_x += deltas[:, 1] * width
    height *= np.exp(deltas[:, 2])
    width *= np.exp(deltas[:, 3])
    # Convert back to y1, x1, y2, x2
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    y2 = y1 + height
    x2 = x1 + width
    return np.stack([y1, x1, y2, x2], axis=1)



def box_refinement(box, gt_box):
    """Compute refinement needed to transform box to gt_box.
    box and gt_box are [N, (y1, x1, y2, x2)]. (y2, x2) is
    assumed to be outside the box.
    """
    box = box.astype(np.float32)
    gt_box = gt_box.astype(np.float32)

    height = box[:, 2] - box[:, 0]
    width = box[:, 3] - box[:, 1]
    center_y = box[:, 0] + 0.5 * height
    center_x = box[:, 1] + 0.5 * width

    gt_height = gt_box[:, 2] - gt_box[:, 0]
    gt_width = gt_box[:, 3] - gt_box[:, 1]
    gt_center_y = gt_box[:, 0] + 0.5 * gt_height
    gt_center_x = gt_box[:, 1] + 0.5 * gt_width

    dy = (gt_center_y - center_y) / height
    dx = (gt_center_x - center_x) / width
    dh = np.log(gt_height / height)
    dw = np.log(gt_width / width)

    return np.stack([dy, dx, dh, dw], axis=1)



def resize_image(image, min_dim=None, max_dim=None, padding=False):
    """
    Resizes an image keeping the aspect ratio.

    min_dim: if provided, resizes the image such that it's smaller
        dimension == min_dim
    max_dim: if provided, ensures that the image longest side doesn't
        exceed this value.
    padding: If true, pads image with zeros so it's size is max_dim x max_dim

    Returns:
    image: the resized image
    window: (y1, x1, y2, x2). If max_dim is provided, padding might
        be inserted in the returned image. If so, this window is the
        coordinates of the image part of the full image (excluding
        the padding). The x2, y2 pixels are not included.
    scale: The scale factor used to resize the image
    padding: Padding added to the image [(top, bottom), (left, right), (0, 0)]
    """
    # Default window (y1, x1, y2, x2) and default scale == 1.
    h, w = image.shape[:2]
    window = (0, 0, h, w)
    scale = 1

    # Scale?
    if min_dim:
        # Scale up but not down
        scale = max(1, min_dim / min(h, w))
    # Does it exceed max dim?
    if max_dim:
        image_max = max(h, w)
        if round(image_max * scale) > max_dim:
            scale = max_dim / image_max
    # Resize image and mask
    if scale != 1:
        image = scipy.misc.imresize(
            image, (round(h * scale), round(w * scale)))
    # Need padding?
    if padding:
        # Get new height and width
        h, w = image.shape[:2]
        top_pad = (max_dim - h) // 2
        bottom_pad = max_dim - h - top_pad
        left_pad = (max_dim - w) // 2
        right_pad = max_dim - w - left_pad
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode='constant', constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
    return image, window, scale, padding



def nms(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


######################
# Language functions #
######################


class Vocabulary(object):
    """Simple vocabulary wrapper. Taken from vse++"""

    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx2wordpos = {} # word with part-of-speech (because a verb and a noun can have same name)
        self.wordpos2idx = {}
        self.idx = 0

    def add_word(self, word, pos):
        word_pos = word + '_' + pos
        if word_pos not in self.wordpos2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx2wordpos[self.idx] = word_pos
            self.wordpos2idx[word_pos] = self.idx
            self.idx += 1

    def words(self):
        """ Return all words in vocab as a list """
        words = []
        for word in self.idx2word.values():
            words.append(word)
        return words

    def wordspos(self):
        wordspos = []
        for wordpos in self.idx2wordpos.values():
            wordspos.append(wordpos)
        return words

    def __call__(self, word):

        #assert word in self.word2idx, 'Word %s not in vocabulary'%word
        if word not in self.word2idx:
            return -1
        else:
            return self.word2idx[word]

    def __len__(self):
        return len(self.wordpos2idx)




def load_pretrained_emb(vocab, emb_dim):
    """From blog.keras.io"""
    embeddings_index = {}
    f = open('/sequoia/data2/jpeyre/datasets/word_embeddings/glove.6B.' + str(emb_dim) + 'd.txt')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype=np.float32)
        embeddings_index[word] = coefs
    f.close()

    embedding_mat = np.zeros((len(vocab), emb_dim), dtype=np.float32)
    for i in range(len(vocab)):
        word = vocab.idx2word[i]
        embedding_v = embeddings_index.get(word)
        if embedding_v is not None:
            embedding_mat[i] = embedding_v

    return embedding_mat




############################################################
#  Miscellaneous
############################################################

def trim_zeros(x):
    """It's common to have tensors larger than the available data and
    pad with zeros. This function removes rows that are all zeros.

    x: [rows, columns].
    """
    assert len(x.shape) == 2
    return x[~np.all(x == 0, axis=1)]


def compute_ap(gt_boxes, gt_class_ids,
               pred_boxes, pred_class_ids, pred_scores,
               iou_threshold=0.5):
    """Compute Average Precision at a set IoU threshold (default 0.5).

    Returns:
    mAP: Mean Average Precision
    precisions: List of precisions at different class score thresholds.
    recalls: List of recall values at different class score thresholds.
    overlaps: [pred_boxes, gt_boxes] IoU overlaps.
    """
    # Trim zero padding and sort predictions by score from high to low
    # TODO: cleaner to do zero unpadding upstream
    gt_boxes = trim_zeros(gt_boxes)
    pred_boxes = trim_zeros(pred_boxes)
    pred_scores = pred_scores[:pred_boxes.shape[0]]
    indices = np.argsort(pred_scores)[::-1]
    pred_boxes = pred_boxes[indices]
    pred_class_ids = pred_class_ids[indices]
    pred_scores = pred_scores[indices]

    # Compute IoU overlaps [pred_boxes, gt_boxes]
    overlaps = compute_overlaps(pred_boxes, gt_boxes)

    # Loop through ground truth boxes and find matching predictions
    match_count = 0
    pred_match = np.zeros([pred_boxes.shape[0]])
    gt_match = np.zeros([gt_boxes.shape[0]])
    for i in range(len(pred_boxes)):
        # Find best matching ground truth box
        sorted_ixs = np.argsort(overlaps[i])[::-1]
        for j in sorted_ixs:
            # If ground truth box is already matched, go to next one
            if gt_match[j] == 1:
                continue
            # If we reach IoU smaller than the threshold, end the loop
            iou = overlaps[i, j]
            if iou < iou_threshold:
                break
            # Do we have a match?
            if pred_class_ids[i] == gt_class_ids[j]:
                match_count += 1
                gt_match[j] = 1
                pred_match[i] = 1
                break

    # Compute precision and recall at each prediction box step
    precisions = np.cumsum(pred_match) / (np.arange(len(pred_match)) + 1)
    recalls = np.cumsum(pred_match).astype(np.float32) / len(gt_match)

    # Pad with start and end values to simplify the math
    precisions = np.concatenate([[0], precisions, [0]])
    recalls = np.concatenate([[0], recalls, [1]])

    # Ensure precision values decrease but don't increase. This way, the
    # precision value at each recall threshold is the maximum it can be
    # for all following recall thresholds, as specified by the VOC paper.
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = np.maximum(precisions[i], precisions[i + 1])

    # Compute mean AP over recall range
    indices = np.where(recalls[:-1] != recalls[1:])[0] + 1
    mAP = np.sum((recalls[indices] - recalls[indices - 1]) *
                 precisions[indices])

    return mAP, precisions, recalls, overlaps


def compute_recall(pred_boxes, gt_boxes, iou):
    """Compute the recall at the given IoU threshold. It's an indication
    of how many GT boxes were found by the given prediction boxes.

    pred_boxes: [N, (y1, x1, y2, x2)] in image coordinates
    gt_boxes: [N, (y1, x1, y2, x2)] in image coordinates
    """
    # Measure overlaps
    overlaps = compute_overlaps(pred_boxes, gt_boxes)
    iou_max = np.max(overlaps, axis=1)
    iou_argmax = np.argmax(overlaps, axis=1)
    positive_ids = np.where(iou_max >= iou)[0]
    matched_gt_boxes = iou_argmax[positive_ids]

    recall = len(set(matched_gt_boxes)) / gt_boxes.shape[0]
    return recall, positive_ids





