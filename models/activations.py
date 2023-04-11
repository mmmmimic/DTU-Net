import torch 
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2
from copy import deepcopy
import math

def fetal_concept_act(x):
    seg_mask = x['seg_mask']
    concept_logit = x['concept_logit']

    # mask out concept preds based on segmentation
    mask = torch.ones_like(concept_logit)
    for b in range(seg_mask.size(0)):
        # cervix
        if ((1 not in seg_mask[b,...]) + (2 not in seg_mask[b,...]) + (3 not in seg_mask[b,...])) >= 3:
            mask[b, [22, 24, 25, 26]] = 0
        if 4 not in seg_mask[b,...]:
            mask[b, 23] = 0
        
        # femur
        if 5 not in seg_mask[b,...]:
            mask[b, [0,1,2,3]] = 0

        # abdomen
        if 6 not in seg_mask[b,...]:
            mask[b, 5] = 0
        if 8 not in seg_mask[b,...]:
            mask[b, 6] = 0
        if 9 not in seg_mask[b,...]:
            mask[b, 7] = 0
        if 7 not in seg_mask[b, ...]:
            mask[b, [9, 10, 11, 12]] = 0
            mask[b, 8] = 0
        if ((6 not in seg_mask[b,...]) + (7 not in seg_mask[b,...]) + (8 not in seg_mask[b,...])) >= 3:
            mask[b, [4, 8]] = 0
            mask[b, [9, 10, 11, 12]] = 0

        # head
        if 10 not in seg_mask[b,...]:
            mask[b, 14] = 0
        if 11 not in seg_mask[b,...]:
            mask[b, 16] = 0
        if 12 not in seg_mask[b,...]:
            mask[b, 15] = 0
        if 13 not in seg_mask[b, ...]:
            mask[b, [18, 19, 20, 21]] = 0
            mask[b, 17] = 0
        if ((10 not in seg_mask[b,...]) + (12 not in seg_mask[b,...]) + (13 not in seg_mask[b,...])) >= 3:
            mask[b, [13, 17]] = 0
            mask[b, [18, 19, 20, 21]] = 0
    
    binary_index = [2, 3, 4, 8, 13, 17, 22, 24]
    quality_index = [0, 1, 5, 6, 7, 9, 10, 11, 12, 14, 15, 16, 18, 19, 20, 21, 23, 25, 26]
    concept_pred = concept_logit.clone()
    concept_pred[:,quality_index] = torch.relu(concept_pred[:, quality_index])
    concept_pred[:,quality_index] = 10 - torch.relu((10-concept_pred[:,quality_index]))
    concept_pred[:,binary_index] = torch.sigmoid(concept_pred[:,binary_index])

    concept_pred = concept_pred*mask
    concept_logit = concept_pred.clone()
    concept_logit[:, 23] = 0 # ignore bladder
    concept_logit[:, [9,10,11,12]] = torch.min(concept_logit[:, [9,10,11,12]], dim=1, keepdim=True)[0] # only take one of the abdomen caliper
    concept_logit[:,quality_index] = concept_logit[:,quality_index]/10
    concept_pred[:,binary_index] = (concept_pred[:,binary_index]>0.5).float() 
    x['concept_mask'] = mask
    x['concept_pred'] = concept_pred
    x['concept_logit'] = concept_logit
    return x

def fetal_concept_act_womask(x):
    seg_mask = x['seg_mask']
    concept_logit = x['concept_logit']

    binary_index = [2, 3, 4, 8, 13, 17, 22, 24]
    quality_index = [0, 1, 5, 6, 7, 9, 10, 11, 12, 14, 15, 16, 18, 19, 20, 21, 23, 25, 26]
    concept_pred = concept_logit.clone()
    concept_pred[:,quality_index] = torch.relu(concept_pred[:, quality_index])
    concept_pred[:,quality_index] = 10 - torch.relu((10-concept_pred[:,quality_index]))
    concept_pred[:,binary_index] = torch.sigmoid(concept_pred[:,binary_index])

    concept_logit = concept_pred.clone()
    concept_logit[:, 23] = 0 # ignore bladder
    concept_logit[:, [9,10,11,12]] = torch.min(concept_logit[:, [9,10,11,12]], dim=1, keepdim=True)[0] # only take one of the abdomen caliper
    concept_logit[:,quality_index] = concept_logit[:,quality_index]/10
    concept_pred[:,binary_index] = (concept_pred[:,binary_index]>0.5).float() 
    x['concept_pred'] = concept_pred
    x['concept_logit'] = concept_logit
    x['concept_mask'] = torch.ones_like(concept_logit)

    return x



def get_ellipse(mask:torch.Tensor, fill_value): # 7, 13
    device = mask.device
    mask = np.asarray(mask.squeeze(-1).detach().cpu().numpy(), dtype=np.float32)
    points = np.nonzero(mask)
    points = np.array(points).T
    point_num = points.shape[0]
    if point_num < 5:
        return torch.from_numpy(mask).unsqueeze(-1).to(device), [(0, 0), (0, 0), (0, 0), (0, 0)]
    (xc, yc), (d1, d2), angle  = cv2.fitEllipse(points) # return ([centeroid coordinate], [length of the semi-major and semi-minor axis], [rotation angle])
    r1, r2 = d1/2, d2/2
    img = np.zeros_like(mask)
    img2 = cv2.ellipse(deepcopy(img), (int(yc), int(xc)), 
                    (int(r2), int(r1)), 
                    -angle, 0, 360, (fill_value), thickness=5)
    left_x = int(xc + math.cos(math.radians(angle))*r1)
    left_y = int(yc + math.sin(math.radians(angle))*r1)
    right_x = int(xc + math.cos(math.radians(angle+180))*r1)
    right_y = int(yc + math.sin(math.radians(angle+180))*r1)
    angle = angle + 90
    top_x = int(xc + math.cos(math.radians(angle))*r2)
    top_y = int(yc + math.sin(math.radians(angle))*r2)
    bottom_x = int(xc + math.cos(math.radians(angle+180))*r2)
    bottom_y = int(yc + math.sin(math.radians(angle+180))*r2)

    mask = torch.from_numpy(img2).unsqueeze(-1).to(device)

    return mask, [(left_x, left_y), (right_x, right_y), (top_x, top_y), (bottom_x, bottom_y)]


def filling_values(img, x, y, patch_size, value):
    x1 = max([0, x-patch_size])
    x2 = min([img.size(0), x+patch_size])
    y1 = max([0, y-patch_size])
    y2 = min([img.size(1), y+patch_size])
    img[x1:x2, y1:y2, ...] = value
    return img

def fetal_caliper_concept(x):
    # add extra segmentation concepts for calipers
    seg_mask = x['seg_mask']
    assign_mtx = x['assign_mtx']
    patch_size = 32
    fill_value = 1.
    caliper_concepts = []

    for b in range(seg_mask.size(0)): # for each image in the batch
        # femur left and right end
        mask = seg_mask[b, ...].unsqueeze(-1)
        left_mask = torch.zeros_like(mask)
        right_mask = torch.zeros_like(mask)
        abdomen_mask = torch.zeros_like(mask)
        bpdn_mask = torch.zeros_like(mask) # top
        bpdf_mask = torch.zeros_like(mask) # bottom
        ofdf_mask = torch.zeros_like(mask) # left or near csp
        ofdo_mask = torch.zeros_like(mask) # right or near thalamus

        if torch.sum(mask==5) >= 100:
            points = torch.nonzero(mask==5)
            index_left, index_right = torch.argmin(points[:,1]), torch.argmax(points[:,1])
            x_left, x_right, y_left, y_right = points[index_left, 0], points[index_right, 0], points[index_left, 1], points[index_right, 1]
            
            left_mask = filling_values(left_mask, x_left, y_left, patch_size, fill_value)
            right_mask = filling_values(right_mask, x_right, y_right, patch_size, fill_value)
        
        elif torch.sum(mask==7) >= 100:
            abdomen_mask, _ = get_ellipse(mask==7, fill_value)
        elif torch.sum(mask==13) >= 100:
            _, points = get_ellipse(mask==13, fill_value)
            left_point, right_point, top_point, bottom_point = points
            bpdn_mask = filling_values(bpdn_mask, top_point[0], top_point[1], patch_size, fill_value)
            bpdf_mask = filling_values(bpdf_mask, bottom_point[0], bottom_point[1], patch_size, fill_value)
            if torch.sum(mask==10) >= 100:
                points = torch.nonzero(mask==10)
                y_c = torch.median(points[:,1])
                if torch.abs(left_point[1] - y_c) < torch.abs(right_point[1] - y_c):
                    tmp = right_point
                    right_point = left_point
                    left_point = tmp
            elif torch.sum(mask==12) >= 100:
                points = torch.nonzero(mask==12)
                y_c = torch.median(points[:,1])
                if torch.abs(left_point[1] - y_c) > torch.abs(right_point[1] - y_c):
                    tmp = right_point
                    right_point = left_point
                    left_point = tmp  
            ofdf_mask = filling_values(ofdf_mask, left_point[0], left_point[1], patch_size, fill_value)
            ofdo_mask = filling_values(ofdo_mask, right_point[0], right_point[1], patch_size, fill_value)            
        caliper_concepts.append(torch.cat((left_mask, right_mask, abdomen_mask, bpdf_mask, bpdn_mask, ofdf_mask, ofdo_mask), dim=-1).unsqueeze(0))
    caliper_concepts = torch.cat(caliper_concepts, dim=0)
    assign_mtx = torch.cat((assign_mtx, caliper_concepts.permute(0,3,1,2)), axis=1)
    
    x['assign_mtx'] = assign_mtx
    return x

# mask = Image.open('/home/manli/3rd_trim_ultrasounds/Trial16/Femur/158_data_proto_RH data 2018_Feb_0211881208_1.2.276.0.26.1.1.1.2.2018.84.28932.8191941.12451840_mask.tif')
# mask = torch.from_numpy(np.asarray(mask)).to('cuda')
# mask = mask.squeeze()
# # ellipse, points = get_ellipse(mask, class_ind=13)
# mask = fetal_caliper_concept({'seg_mask': mask.unsqueeze(0), 'assign_mtx':torch.nn.functional.one_hot(mask.long().unsqueeze(0), num_classes=14).permute(0,3,1,2)})['assign_mtx']

# mask = mask.squeeze().detach().cpu().numpy()