from locale import normalize
from multiprocessing import reduction
import pdb
from turtle import pd
import numpy as np
import torch.nn as nn
import torch
import random
from utils.losses import mask_DiceLoss
from scipy.ndimage import distance_transform_edt as distance
from skimage import segmentation as skimage_seg

DICE = mask_DiceLoss(nclass=2)
CE = nn.CrossEntropyLoss(reduction='none')

def context_mask(img, mask_ratio):
    batch_size, channel, img_x, img_y, img_z = img.shape[0],img.shape[1],img.shape[2],img.shape[3],img.shape[4]
    loss_mask = torch.ones(batch_size, img_x, img_y, img_z).cuda()
    mask = torch.ones(img_x, img_y, img_z).cuda()
    patch_pixel_x, patch_pixel_y, patch_pixel_z = int(img_x*mask_ratio), int(img_y*mask_ratio), int(img_z*mask_ratio)
    w = np.random.randint(0, 112 - patch_pixel_x)
    h = np.random.randint(0, 112 - patch_pixel_y)
    z = np.random.randint(0, 80 - patch_pixel_z)
    mask[w:w+patch_pixel_x, h:h+patch_pixel_y, z:z+patch_pixel_z] = 0
    loss_mask[:, w:w+patch_pixel_x, h:h+patch_pixel_y, z:z+patch_pixel_z] = 0
    return mask.long(), loss_mask.long()

def random_mask(img):
    batch_size, channel, img_x, img_y, img_z = img.shape[0],img.shape[1],img.shape[2],img.shape[3],img.shape[4]
    loss_mask = torch.ones(batch_size, img_x, img_y, img_z).cuda()
    mask = torch.ones(img_x, img_y, img_z).cuda()
    patch_pixel_x, patch_pixel_y, patch_pixel_z = int(img_x*2/3), int(img_y*2/3), int(img_z*2/3)
    mask_num = 27
    mask_size_x, mask_size_y, mask_size_z = int(patch_pixel_x/3)+1, int(patch_pixel_y/3)+1, int(patch_pixel_z/3)
    size_x, size_y, size_z = int(img_x/3), int(img_y/3), int(img_z/3)
    for xs in range(3):
        for ys in range(3):
            for zs in range(3):
                w = np.random.randint(xs*size_x, (xs+1)*size_x - mask_size_x - 1)
                h = np.random.randint(ys*size_y, (ys+1)*size_y - mask_size_y - 1)
                z = np.random.randint(zs*size_z, (zs+1)*size_z - mask_size_z - 1)
                mask[w:w+mask_size_x, h:h+mask_size_y, z:z+mask_size_z] = 0
                loss_mask[:, w:w+mask_size_x, h:h+mask_size_y, z:z+mask_size_z] = 0
    return mask.long(), loss_mask.long()

def concate_mask(img):
    batch_size, channel, img_x, img_y, img_z = img.shape[0],img.shape[1],img.shape[2],img.shape[3],img.shape[4]
    loss_mask = torch.ones(batch_size, img_x, img_y, img_z).cuda()
    mask = torch.ones(img_x, img_y, img_z).cuda()
    z_length = int(img_z * 8 / 27)
    z = np.random.randint(0, img_z - z_length -1)
    mask[:, :, z:z+z_length] = 0
    loss_mask[:, :, :, z:z+z_length] = 0
    return mask.long(), loss_mask.long()

def find_min_rect(tensor):
    indices = torch.where(tensor == 1)
    min_x = indices[0].min().item()
    max_x = indices[0].max().item()
    min_y = indices[1].min().item()
    max_y = indices[1].max().item()
    min_z = indices[2].min().item()
    max_z = indices[2].max().item()
    return min_x, max_x, min_y, max_y, min_z, max_z

def create_mask(img_a, img_b, lab_a, lab_b):
    batch_size, channel, img_x, img_y, img_z = img_a.shape[0], img_a.shape[1], img_a.shape[2], img_a.shape[3], img_a.shape[4]
    mask = torch.ones(batch_size, img_x, img_y, img_z).cuda()
    mask_ratio = 2 / 3
    s_x, s_y, s_z = int(img_x * mask_ratio), int(img_y * mask_ratio), int(img_z * mask_ratio)
    # a_start_x = s_x
    # a_start_y = s_y
    # a_start_z = s_z
    # b_start_x = s_x
    # b_start_y = s_y
    # b_start_z = s_z
    for i in range(batch_size):
        a_min_rect = find_min_rect(lab_a[i,:,:,:])
        b_min_rect = find_min_rect(lab_b[i,:,:,:])
        a_x = (a_min_rect[1] - a_min_rect[0]) + 1
        a_y = (a_min_rect[3] - a_min_rect[2]) + 1
        a_z = (a_min_rect[5] - a_min_rect[4]) + 1
        b_x = (b_min_rect[1] - b_min_rect[0]) + 1
        b_y = (b_min_rect[3] - b_min_rect[2]) + 1
        b_z = (b_min_rect[5] - b_min_rect[4]) + 1
        f_x = min(a_x, b_x)
        f_y = min(a_y, b_y)
        f_z = min(a_z, b_z)
        f_x = min(f_x, s_x)
        f_y = min(f_y, s_y)
        f_z = min(f_z, s_z)
        center_ax = (a_min_rect[0] + a_min_rect[1]) / 2
        center_ay = (a_min_rect[2] + a_min_rect[3]) / 2
        center_az = (a_min_rect[4] + a_min_rect[5]) / 2
        center_bx = (b_min_rect[0] + b_min_rect[1]) / 2
        center_by = (b_min_rect[2] + b_min_rect[3]) / 2
        center_bz = (b_min_rect[4] + b_min_rect[5]) / 2
        a_start_x = int(center_ax - (f_x - 1) / 2)
        a_start_y = int(center_ay - (f_y - 1) / 2)
        a_start_z = int(center_az - (f_z - 1) / 2)
        b_start_x = int(center_bx - (f_x - 1) / 2)
        b_start_y = int(center_by - (f_y - 1) / 2)
        b_start_z = int(center_bz - (f_z - 1) / 2)
        img_a[i, :, a_start_x:a_start_x+f_x, a_start_y:a_start_y+f_y, a_start_z:a_start_z+f_z] = img_b[i, :, b_start_x:b_start_x+f_x, b_start_y:b_start_y+f_y, b_start_z:b_start_z+f_z]
        mask[i, a_start_x:a_start_x+f_x, a_start_y:a_start_y+f_y, a_start_z:a_start_z+f_z] = 0
        lab_a[i, a_start_x:a_start_x+f_x, a_start_y:a_start_y+f_y, a_start_z:a_start_z+f_z] = lab_b[i, b_start_x:b_start_x+f_x, b_start_y:b_start_y+f_y, b_start_z:b_start_z+f_z]
        # if state == 0:
        #     lab_a[i, a_start_x:a_start_x+f_x, a_start_y:a_start_y+f_y, a_start_z:a_start_z+f_z] = lab_b[i, b_start_x:b_start_x+f_x, b_start_y:b_start_y+f_y, b_start_z:b_start_z+f_z]
        # elif state == 1:
        #     lab_a[i, a_start_x:a_start_x+f_x, a_start_y:a_start_y+f_y, a_start_z:a_start_z+f_z] = plab_b[i, b_start_x:b_start_x+f_x, b_start_y:b_start_y+f_y, b_start_z:b_start_z+f_z]
        # elif state == 2:
        #     plab_a[i, a_start_x:a_start_x+f_x, a_start_y:a_start_y+f_y, a_start_z:a_start_z+f_z] = lab_b[i, b_start_x:b_start_x+f_x, b_start_y:b_start_y+f_y, b_start_z:b_start_z+f_z]
        #     lab_a[i] = plab_a[i]
        # else:
        #     plab_a[i, a_start_x:a_start_x+f_x, a_start_y:a_start_y+f_y, a_start_z:a_start_z+f_z] = plab_b[i, b_start_x:b_start_x+f_x, b_start_y:b_start_y+f_y, b_start_z:b_start_z+f_z]
        #     lab_a[i] = plab_a[i]
    return img_a, lab_a, mask

def mix_loss(net3_output, img_l, patch_l, mask, mask_p=None, mask_n=None, l_weight=1.0, u_weight=0.5, unlab=False):
    img_l, patch_l = img_l.type(torch.int64), patch_l.type(torch.int64)
    image_weight, patch_weight = l_weight, u_weight
    if unlab:
        image_weight, patch_weight = u_weight, l_weight
    patch_mask = 1 - mask
    if mask_p is not None:
        mask = mask + mask_p
    if mask_n is not None:
        patch_mask = patch_mask + mask_n
    dice_loss = DICE(net3_output, img_l, mask) * image_weight
    dice_loss += DICE(net3_output, patch_l, patch_mask) * patch_weight
    loss_ce = image_weight * (CE(net3_output, img_l) * mask).sum() / (mask.sum() + 1e-16)
    loss_ce += patch_weight * (CE(net3_output, patch_l) * patch_mask).sum() / (patch_mask.sum() + 1e-16)
    loss = (dice_loss + loss_ce) / 2
    return loss

def mix_loss_sep(net3_output, img_l, patch_l, mask, mask_p=None, l_weight=1.0, u_weight=0.5, unlab=False):
    img_l, patch_l = img_l.type(torch.int64), patch_l.type(torch.int64)
    image_weight, patch_weight = l_weight, u_weight
    if unlab:
        image_weight, patch_weight = u_weight, l_weight
    if mask_p is None:
        patch_mask = 1 - mask
    else:
        patch_mask = mask_p
    dice_loss = DICE(net3_output, img_l, mask) * image_weight
    dice_loss += DICE(net3_output, patch_l, patch_mask) * patch_weight
    loss_ce = image_weight * (CE(net3_output, img_l) * mask).sum() / (mask.sum() + 1e-16)
    loss_ce += patch_weight * (CE(net3_output, patch_l) * patch_mask).sum() / (patch_mask.sum() + 1e-16)
    loss = (dice_loss + loss_ce) / 2
    return loss

def mix_loss_new(net3_output, img_l, patch_l, mask, mask_p=None, mask_n=None, l_weight=1.0, u_weight=0.5, unlab=False):
    img_l, patch_l = img_l.type(torch.int64), patch_l.type(torch.int64)
    image_weight, patch_weight = l_weight, u_weight
    if unlab:
        image_weight, patch_weight = u_weight, l_weight
    patch_mask = 1 - mask
    with torch.no_grad():
        dice_loss_ni = DICE(net3_output, img_l, mask) * image_weight
        dice_loss_np = DICE(net3_output, patch_l, patch_mask) * patch_weight
        loss_ce_ni = image_weight * (CE(net3_output, img_l) * mask).sum() / (mask.sum() + 1e-16)
        loss_ce_np = patch_weight * (CE(net3_output, patch_l) * patch_mask).sum() / (patch_mask.sum() + 1e-16)
    if mask_p is not None:
        mask = mask + mask_p
    if mask_n is not None:
        patch_mask = patch_mask + mask_n
    dice_loss_i = DICE(net3_output, img_l, mask) * image_weight
    dice_loss_p = DICE(net3_output, patch_l, patch_mask) * patch_weight
    loss_ce_i = image_weight * (CE(net3_output, img_l) * mask).sum() / (mask.sum() + 1e-16)
    loss_ce_p = patch_weight * (CE(net3_output, patch_l) * patch_mask).sum() / (patch_mask.sum() + 1e-16)
    with torch.no_grad():
        dni = dice_loss_ni / dice_loss_i
        dnp = dice_loss_np / dice_loss_p
        cni = loss_ce_ni / loss_ce_i
        cnp = loss_ce_np / loss_ce_p
    loss = (dice_loss_i * dni + dice_loss_p * dnp + loss_ce_i * cni + loss_ce_p * cnp) / 2
    return loss

def sup_mask_loss(output, label, mask):
    label = label.type(torch.int64)
    dice_loss = DICE(output, label, mask)
    loss_ce = (CE(output, label) * mask).sum() / (mask.sum() + 1e-16)
    loss = (dice_loss + loss_ce) / 2
    return loss

def sup_loss(output, label):
    label = label.type(torch.int64)
    dice_loss = DICE(output, label)
    loss_ce = torch.mean(CE(output, label))
    loss = (dice_loss + loss_ce) / 2
    return loss

@torch.no_grad()
def update_ema_variables(model, ema_model, alpha):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_((1 - alpha) * param.data)

@torch.no_grad()
def update_ema_students(model1, model2, ema_model, alpha):
    for ema_param, param1, param2 in zip(ema_model.parameters(), model1.parameters(), model2.parameters()):
        ema_param.data.mul_(alpha).add_(((1 - alpha)/2) * param1.data).add_(((1 - alpha)/2) * param2.data)

@torch.no_grad()
def parameter_sharing(model, ema_model):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data = param.data

class BBoxException(Exception):
    pass

def get_non_empty_min_max_idx_along_axis(mask, axis):
    """
    Get non zero min and max index along given axis.
    :param mask:
    :param axis:
    :return:
    """
    if isinstance(mask, torch.Tensor):
        # pytorch is the axis you want to get
        nonzero_idx = (mask != 0).nonzero()
        if len(nonzero_idx) == 0:
            min = max = 0
        else:
            max = nonzero_idx[:, axis].max()
            min = nonzero_idx[:, axis].min()
    elif isinstance(mask, np.ndarray):
        nonzero_idx = (mask != 0).nonzero()
        if len(nonzero_idx[axis]) == 0:
            min = max = 0
        else:
            max = nonzero_idx[axis].max()
            min = nonzero_idx[axis].min()
    else:
        raise BBoxException("Wrong type")
    max += 1
    return min, max


def get_bbox_3d(mask):
    """ Input : [D, H, W] , output : ((min_x, max_x), (min_y, max_y), (min_z, max_z))
    Return non zero value's min and max index for a mask
    If no value exists, an array of all zero returns
    :param mask:  numpy of [D, H, W]
    :return:
    """
    assert len(mask.shape) == 3
    min_z, max_z = get_non_empty_min_max_idx_along_axis(mask, 2)
    min_y, max_y = get_non_empty_min_max_idx_along_axis(mask, 1)
    min_x, max_x = get_non_empty_min_max_idx_along_axis(mask, 0)

    return np.array(((min_x, max_x),
                     (min_y, max_y),
                     (min_z, max_z)))

def get_bbox_mask(mask):
    batch_szie, x_dim, y_dim, z_dim = mask.shape[0], mask.shape[1], mask.shape[2], mask.shape[3]
    mix_mask = torch.ones(batch_szie, 1, x_dim, y_dim, z_dim).cuda()
    for i in range(batch_szie):
        curr_mask = mask[i, ...].squeeze()
        (min_x, max_x), (min_y, max_y), (min_z, max_z) = get_bbox_3d(curr_mask)
        mix_mask[i, :, min_x:max_x, min_y:max_y, min_z:max_z] = 0
    return mix_mask.long()

