from joblib import Parallel, delayed
from scipy.spatial.distance import cdist
from vcdist_funcs import vc_dis_paral, vc_dis_paral_full
import time
import pickle
import os
from config_initialization import vc_num, dataset, data_path, cat_test, device_ids, Astride, Apad, Arf,vMF_kappa, layer,init_path, dict_dir, sim_dir, extractor, nn_type
from Code.helpers import getImg, imgLoader, Imgset, myresize
from DataLoader import KINS_Compnet_Train_Dataset
from torch.utils.data import DataLoader
import numpy as np
import math
import torch

categories = ['cyclist', 'car', 'tram', 'truck', 'van']

paral_num = 10
nimg_per_cat = 2000
height_threshold = 75
imgs_par_cat =np.zeros(len(categories))


print('max_images {}'.format(nimg_per_cat))

if not os.path.exists(sim_dir):
	os.makedirs(sim_dir)

#############################
# BEWARE THIS IS RESET TO LOAD OLD VCS AND MODEL
#############################
with open('../meta/init_{}/dictionary_vgg_kinsv/dictionary_{}_{}_kappa{}.pickle'.format(nn_type, layer,vc_num, vMF_kappa), 'rb') as fh:
	centers = pickle.load(fh)
##HERE
bool_pytorch = True

for category in categories:
	cat_idx = categories.index(category)
	print('{} / {}'.format(c