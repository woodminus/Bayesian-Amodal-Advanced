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
        for j in range(w):

            within_inner_bbox = False
            within_outer_bbox = False

            for b in range(bbox.shape[0]):

                inner_box = inner_bbox[b]
                within_inner_bbox = within_inner_bbox or (i >= inner_box[0] and i < inner_box[2] and j >= inner_box[1] and j < inner_box[3])

                outer_box = outer_bbox[b]
                within_outer_bbox = within_outer_bbox or (i >= outer_box[0] and i < outer_box[2] and j >= outer_box[1] and j < outer_box[3])

            if within_outer_bbox and not within_inner_bbox:
                context_feat.append((features[i][j] / np.sqrt(np.sum(features[i][j] ** 2) + 1e-10)).tolist())
                #demo[i][j] = 1
    return context_feat#, demo


def learn_context_feature(category, inner_bound=16, outer_bound=128, percentage_for_clustering=.1, max_num=100000, num_cluster=10):

    # Stage 1: Collect features that have receptive field outside of object bounding boxes
    print('==========Class: {}=========='.format(category))
    print('Stage 1: Feature Extraction')
    storage_dir = init_dir + 'context_features_meta_{}/'.format(dataset_train)
    if not os.path.exists(storage_dir):
        os.makedirs(storage_dir)

    storage_file = storage_dir + '{}_context_features_[{} - {}].pickle'.format(category, inner_bound, outer_bound)

    if not os.path.exists(storage_file):

        feat_vec = []

        if dataset_train == 'pascal3d+':

            image_files, mask_files, labels, bboxs = get_pascal3d_data(cats=[category], train=True, single_obj=False)
            data_set = Multi_Object_Loader(image_files, mask_files, labels, bboxs, resize=True, min_size=200, max_size=3000, demo_img_return=True)

        elif dataset_train == 'kins':
            data_set = KINS_Dataset(category_list=[category], dataType='train', occ=[0, 1],
                                    height_thrd=75, amodal_height=False, frac=1.0,
                                    demo_img_return=True)

        data_loader = DataLoader(dataset=data_set, batch_size=1, shuffle=False)

        N = data_loader.__len__()

        for ii, data in enumerate(data_loader):
            if dataset_train == 'pascal3d+':
                input, label, bbox, gt_mask, scale, demo_img, img_path = data

            elif dataset_train == 'kins':
                input, gt_labels, bbox, gt_amodal_bbox, gt_inmodal_segmentation, gt_amodal_segmentation, gt_occ, demo_img, img_path = 