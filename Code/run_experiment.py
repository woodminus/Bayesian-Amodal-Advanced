from configs import device_ids, dataset_train, dataset_eval, nn_type, vc_num, K, vMF_kappa, context_cluster, layer, exp_dir, categories, feature_num, TABLE_NUM, MODEL_TYPE
from configs import *
from model import get_compnet_head
from DataLoader import Occ_Veh_Dataset, KINS_Dataset, COCOA_Dataset
from scipy import interpolate
from torch.utils.data import DataLoader
from util import roc_curve, rank_perf, visualize, draw_box, visualize_multi, visualize_mask, calc_iou, find_max_iou
import copy
import sys
import torch
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


def make_three_dimensional_demo(pixel_cls, pixel_cls_score):

    occ_label = 0
    fg_label = 1
    bg_label = 2

    try:
        occ_range = [np.min(pixel_cls_score[pixel_cls == occ_label]), np.max(pixel_cls_score[pixel_cls == occ_label])]
        if occ_range[1] - occ_range[0] == 0:
            occ_range = [0, 1]
    except:
        occ_range = [0, 1]

    try:
        fg_range = [np.min(pixel_cls_score[pixel_cls == fg_label]), np.max(pixel_cls_score[pixel_cls == fg_label])]
        if fg_range[1] - fg_range[0] == 0:
            fg_range = [0, 1]
    except:
        fg_range = [0, 1]

    try:
        bg_range = [np.min(pixel_cls_score[pixel_cls == bg_label]), np.max(pixel_cls_score[pixel_cls == bg_label])]
        if bg_range[1] - bg_range[0] == 0:
            bg_range = [0, 1]
    except:
        bg_range = [0, 1]

    # all_min = min(occ_range[0], fg_range[0], bg_range[0] )
    # all_max = max(occ_range[1], fg_range[1], bg_range[1] )
    # occ_range = [all_min, all_max]
    # fg_range = [all_min, all_max]
    # bg_range = [all_min, all_max]

    # treat an rbg image as three layers heatmap

    occ_layer = ( (pixel_cls == occ_label).astype(float) * (pixel_cls_score - occ_range[0]) / (occ_range[1] - occ_range[0]) * 255 ).astype(int)[:, :, np.newaxis]
    fg_layer  = ( (pixel_cls == fg_label).astype(float) * (pixel_cls_score - fg_range[0]) / (fg_range[1] - fg_range[0]) * 255 ).astype(int)[:, :, np.newaxis]
    bg_layer  = ( (pixel_cls == bg_label).astype(float) * (pixel_cls_score - bg_range[0]) / (bg_range[1] - bg_range[0]) * 255 ).astype(int)[:, :, np.newaxis]

    # cv2.imwrite('temp_pred_check_o.jpg', occ_layer)
    # cv2.imwrite('temp_pred_check_f.jpg', fg_layer)
    # cv2.imwrite('temp_pred_check_c.jpg', bg_layer)

    img = np.concatenate((fg_layer, bg_layer, occ_layer), axis=2)

    return img

def print_(str, file=None):
    print(str)
    if file:
        print(str, file=file)

def make_demo(data_loader, rank, img_index, obj_index, exp_set_up, out_dir=''):
    mask_type, input_bbox_type, input_gt_label, bmask_thrd = exp_set_up

    out_dir_m = out_dir + '{}/'.format(mask_type)
    if not os.path.exists(out_dir_m):
        os.mkdir(out_dir_m)

    out_dir_p = out_dir + 'pixel_cls/'
    if not os.path.exists(out_dir_p):
        os.mkdir(out_dir_p)

    input_label = None
    input_bbox = None
    gt_seg = None

    input_tensor, gt_labels, gt_inmodal_bbox, gt_amodal_bbox, gt_inmodal_segmentation, gt_amodal_segmentation, gt_occ, demo_img, img_path, failed = data_loader.dataset.__getitem__(img_index)


    gt_inmodal_bbox, gt_amodal_bbox = torch.tensor(gt_inmodal_bbox), torch.tensor(gt_amodal_bbox)

    if input_bbox_type == 'inmodal':
        input_bbox = copy.deepcopy(gt_inmodal_bbox)
    elif input_bbox_type == 'amodal':
        input_bbox = copy.deepcopy(gt_amodal_bbox)

    if input_gt_label:
        input_label = gt_labels.copy()
    try:
        with torch.no_grad():
            c, h, w = input_tensor.shape
            input_tensor = torch.tensor(input_tensor).view(1, c, h, w)

            pred_scores, pred_confidence, pred_amodal_bboxes, pred_segmentations = net(org_x=input_tensor.cuda(device_ids[0]),
                                                                                       bboxes=input_bbox[obj_index:obj_index+1],
                                                                                       bbox_type=input_bbox_type,
                                                                                       input_label=input_label[obj_index:obj_index+1])
    except:
        print('error')

    pred_segmentation = pred_segmentations[0]
    pixel_cls_b = pred_segmentation['pixel_cls']
    pixel_cls_score = pred_segmentation['pixel_cls_score']

    # box = np.copy(gt_amodal_bbox[obj_index])                      #TODO
    box = np.copy(pred_amodal_bboxes[0])
    height_box = box[2] - box[0]
    width_box = box[3] - box[1]

    box[0] = max(0, box[0] - 0.1 * height_box)
    box[1] = max(0, box[1] - 0.1 * width_box)
    box[2] = min(input_tensor.shape[2], box[2] + 0.1 * height_box)
    box[3] = min(input_tensor.shape[3], box[3] + 0.1 * width_box)
    box = box.astype(int)

    copy_img = draw_box(demo_img.copy(), gt_inmodal_bbox[obj_index], color=(255, 0, 0), thick=2)
    copy_img = draw_box(copy_img, pred_amodal_bboxes[0], color=(0, 255, 0), thick=2)
    demo_img = copy_img[box[0]:box[2], box[1]:box[3], :]
    pixel_cls_img = make_three_dimensional_demo(pixel_cls=pixel_cls_b, pixel_cls_score=pixel_cls_score)[box[0]:box[2],
        box[1]:box[3], :]

    if mask_type == 'amodal':
        gt_seg = gt_amodal_segmentation[obj_index]

    elif mask_type == 'inmodal':
        gt_seg = gt_inmodal_segmentation[obj_index]

    elif mask_type == 'occ':
        gt_seg = (gt_amodal_segmentation[obj_index] - gt_inmodal_segmentation[obj_index] > 0).astype(float)

    pred_seg = (pred_segmentation[mask_type] >= bmask_thrd).astype(float)

    visualize_multi(demo_im