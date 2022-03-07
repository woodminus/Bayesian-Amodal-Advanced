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
            for suf in ['_general', '_ijcv']:       # the first clutter is the general one used for classification, the rest is used for segmentation
                clutter = np.concatenate((clutter, np.load( meta_dir + 'ML_{}/{}_{}_clutter_model{}.npy'.format(nn_type, nn_type, layer, suf)) ), axis=0)
    except:
        error_message('Failed to load Clutter Models.')

    clutter = clutter[:, :, np.newaxis, np.newaxis]

    #remove clutter TODO
    # clutter = np.zeros(clutter.shape)
    clutter = torch.from_numpy(clutter).type(torch.FloatTensor)

    return clutter

def get_mixture_models(dim_reduction=True, tag='_it2', dataset_override=None):
    if dataset_override != None:
        dataset = dataset_override
    else:
        dataset = dataset_train

    if tag == '_cross_domain':
        cats = categories['kinsv']
    else:
        cats = categories['train']
    FG_Models = []
    FG_prior = []
    CNTXT_Models = []
    CNTXT_prior = []
    for category in cats:
        load_path = meta_dir + 'ML_{}/mix_model_vmf_{}_EM_all_context{}/mmodel_{}_K{}_FEATDIM512_{}_specific_view_{}.pickle'.format(nn_type, dataset, tag, category, K, layer, context_cluster)
        try:
            alpha, beta, prior = np.load(load_path, allow_pickle=True)
        except:
            error_message('Failed to load Mixture Model: {} \nInput filename: {}'.format(category.upper(), load_path))
            FG_Models.append(None)
            FG_prior.append(None)
            CNTXT_Models.append(None)
            CNTXT_prior.append(None)
            continue

        mix_fg = np.array(alpha)
        mix_context = np.array(beta)
        prior_fg = np.array(prior)[:, 0, :, :]
        prior_context = np.array(prior)[:, 1, :, :]

        # Reduce dimensions of the mixture model since most of the boundary regions only sampled one or two images during model building
        if dim_reduction:
            old_dim = prior_fg.shape
            prior_whole = prior_fg + prior_context
            h_cut=1
            w_cut=1
            while np.sum(prior_whole[:, h_cut:-h_cut, :].reshape(-1, 1)) / np.sum(prior_whole.reshape(-1, 1)) > 0.995:
                h_cut += 1

            while np.sum(prior_whole[:, :, w_cut:-w_cut].reshape(-1, 1)) / np.sum(prior_whole.reshape(-1, 1)) > 0.995:
                w_cut += 1

            mix_fg = mix_fg[:, :, w_cut:-w_cut, h_cut:-h_cut]
            mix_context = mix_context[:, :, w_cut:-w_cut, h_cut:-h_cut]
            prior_fg = prior_fg[:, h_cut:-h_cut, w_cut:-w_cut]
            prior_context = prio