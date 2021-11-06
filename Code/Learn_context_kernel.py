from configs import init_dir, context_cluster, categories, dataset_train
from configs import *
from DataLoader import get_pascal3d_data, Multi_Object_Loader, KINS_Dataset
from model import get_backbone_extractor
from vMFMM import vMFMM
from torch.utils.data import DataLoader
from util import visualize
import random as rm

'''
Learn class specific contextual features
Code Status - Currently Active 6/26
'''


# this method restrict the sampling region to between the inner and outer bound
# Applied L2 normalization
def mask_features(features, bbox, img_shape, inner_bound, outer_bound):
    h, w, c = features.shape

    inner_bbox = bbox.copy()
    inner_bbox[:, 0] = (bbox[:, 0] - inner_bound) / img_shape[0] * h
    inner_bbox[:, 1] = (bbox[:, 1] - inner_bound) / img_shape[1] * w
    inner_bbox[:, 2] = (bbox[:, 2] + inner_bound) / img_shape[0] * h
    inner_bbox[:, 3] = (bbox[:, 3] + inner_bound) / img_shape[1] * w

    outer_bbox = bbox.copy()
    outer_bbox[:, 0] = (bbox[:, 0] - outer_bound) / img_shape[0] * h
    outer_bbox[:, 1] = (bbox[:, 1] - outer_bound) / img_shape[1] * w
    outer_bbox[:, 2] = (bbox[:, 2] + outer_bound) / img_shape[0] * h
    outer_bbox[:, 3] = (bbox[:, 3] + outer_bound) / img_shape[1] * w

    context_feat = []
    #demo = np.zeros((h, w))

    for i in range(h):
        fo