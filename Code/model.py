import torch
import torch.nn as nn
import torch.cuda
import torchvision.models as models

from configs import device_ids, dataset_train, dataset_eval, nn_type, vc_num, K, vMF_kappa, context_cluster, layer, meta_dir, categories, feature_num, rpn_configs, TABLE_NUM
from configs import *
from Net import Net


def vgg16(layer):
    net = models.vgg16(pretrained=True)
    if layer == 'pool5':
        num_layers = 31
    elif layer == 'pool4':
        num_layers = 24
    elif layer == 'pool3':
        num_layers = 17
    model = nn.Sequential()
    features = nn.Sequential()
    for i in range(0, num_layers):
        features.add_module('{}'.format(i), net.features[i])
    model.add_module('features', features)
    return model

def resnext(layer):
    extractor = nn.Sequential()
    net = models.resnext50_32x4d(pretrained=True)
    if layer == 'last':
        extractor.add_module('0', net.conv1)
        extractor.add_module('1', net.bn1)
        extractor.add_module('2', net.relu)
        extractor.add_module('3', net.maxpool)
        extractor.add_module('4', net.layer1)
        extractor.add_module('5', net.layer2)
        extractor.add_module('6', net.layer3)
        extractor.add_module('7', net.layer4)
    elif layer == 'second':
        extractor.add_module('0', net.conv1)
        extractor.add_module('1', net.bn1)
        extractor.add_module('2', net.relu)
        extractor.add_module('3', net.maxpool)
        extractor.add_module('4', net.layer1)
        extractor.add_module('5', net.layer2)
        extractor.add_module('6', net.layer3)
    else:
        extractor = []
    return extractor


# return backbone extractor based on nn_type and layer in configs
def get_backbone_extractor():
    if nn_type == 'vgg':
        return vgg16(layer).cuda(device_ids[0])

    if nn_type == 'resnext':
        return resnext(layer).cuda(device_ids[0]).eval()

    error_message('Failed to get backbone extractor. \nInput nn_type: {}'.format(nn_type))


# return visual concept centers
def get_vc(dataset_override=None):
    if dataset_override != None:
        dataset = dataset_override
        vMF_kappa_ = vMF_kappas['{}_{}_{}'.format(nn_type, layer, dataset_override)]
    else:
        dataset = dataset_train
        vMF_kappa_ = vMF_kappa
    vc = np.zeros((vc_num, feature_num))

    file_name = meta_dir + 'ML_{0}/dictionary_{0}_{1}/dictionary_{2}_{3}_kappa{4}.pickle'.format(nn_type, dataset, layer, vc_num, vMF_kappa_)
    try:
        vc = np.load(file_name, allow_pickle=True)
    except:
        error_message('Failed to load VC. \nInput filename: {}'.format(file_name))
    
    vc = vc[:, :, np.newaxis, np.newaxis]
    vc = torch.from_numpy(vc).type(torch.FloatTensor)
    return vc.cuda(device_ids[0])


# return context cluster centers
def get_context(dataset_override=None):
    if dataset_override != None:
        dataset = dataset_override
    else:
        dataset = dataset_train
    context = np.zeros((0, feature_num))
    for category in categories['train']:
        file_name = meta_dir + 'ML_{}/context_kernel_{}_{}/{}_{}.npy'.format(nn_type, layer, dataset, category, context_cluster)
        try:
            context = np.concatenate((context, np.load(file_name)), axis=0)
        except:
            continue

    context = context[:, :, np.newaxis, np.newaxis]
    context = torch.from_numpy(context).type(torch.FloatTensor)
    return context.cuda(device_ids[0])

def get_clutter_models():
    clutter = np.zeros((0, vc_num))

    try:
        if nn_type == 'vgg':
            clutter = np.load(meta_dir + 'ML_{}/CLUTTER_MODEL_POOL4.pkl'.format(nn_type, nn_type, layer), allow_pickle=True)
            for i in range(clutter.shape[0]):
                clutter[i] = clutter[i] / clutter[i].sum()

        elif nn_type == 'resnext':
            for suf in ['_general', '_ijcv']:       # the first clutter is the general one 