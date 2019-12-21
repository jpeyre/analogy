from __future__ import division
import torch
import torch.nn as nn


class CroppedBoxCoordinates(nn.Module):
    def __init__(self):
        super(CroppedBoxCoordinates, self).__init__()

    def forward(self, pair_objects):

        """
        Get cropped box coordinates [x1,y1,w1,h1,x2,y2,w2,h2] as if the image was the union of boxes and renormalize by union of box area
        """

        subject_boxes = pair_objects[:,0,:4]
        object_boxes = pair_objects[:,1,:4]
        union_boxes = torch.cat((torch.min(subject_boxes[:,:2], object_boxes[:,:2]), torch.max(subject_boxes[:,2:4], object_boxes[:,2:4])),1)
        width_union = union_boxes[:,2]-union_boxes[:,0]+1
        height_union = union_boxes[:,3]-union_boxes[:,1]+1
        area_union = width_union*height_union
        area_union = area_union.sqrt()
        area_union = area_union.unsqueeze(1).expand_as(subject_boxes)

        # Get x,y,w,h in union box coordinates system: copy input
        subject_boxes_trans = subject_boxes.clone()
        object_boxes_trans = object_boxes.clone()
        subject_boxes_trans[:,0].data.copy_(subject_boxes_trans.data[:,0]-union_boxes.data[:,0])
        subject_boxes_trans[:,2].data.copy_(subject_boxes_trans.data[:,2]-union_boxes.data[:,0])
        subject_boxes_trans[:,1].data.copy_(subject_boxes_trans.data[:,1]-union_boxes.data[:,1])
        subject_boxes_trans[:,3].data.copy_(subject_boxes_trans.data[:,3]-union_boxes.data[:,1])
        object_boxes_trans[:,0].data.copy_(object_boxes_trans.data[:,0]-union_boxes.data[:,0])
        object_boxes_trans[:,2].data.copy_(object_boxes_trans.data[:,2]-union_boxes.data[:,0])
        object_boxes_trans[:,1].data.copy_(object_boxes_trans.data[:,1]-union_boxes.data[:,1])
        object_boxes_trans[:,3].data.copy_(object_boxes_trans.data[:,3]-union_boxes.data[:,1])

        # Get x,y,w,h
        subject_boxes_trans[:,2:4].data.copy_(subject_boxes_trans.data[:,2:4]-subject_boxes_trans.data[:,:2]+1)
        object_boxes_trans[:,2:4].data.copy_(object_boxes_trans.data[:,2:4]-object_boxes_trans.data[:,:2]+1)
        

        # Renormalize by union box size
        subject_boxes_trans = subject_boxes_trans.mul(1/area_union)
        object_boxes_trans = object_boxes_trans.mul(1/area_union)

        output = torch.cat([subject_boxes_trans, object_boxes_trans],1)

        return output

