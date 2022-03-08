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

    occ_l