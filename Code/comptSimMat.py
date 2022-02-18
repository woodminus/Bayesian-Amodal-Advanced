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
	print('{} / {}'.format(cat_idx,len(categories)))

	imgset = KINS_Compnet_Train_Dataset(category=category, height_thrd=height_threshold)
	data_loader = DataLoader(dataset=imgset, batch_size=1, shuffle=False)
	N= min(data_loader.__len__(), nimg_per_cat)

	savename = os.path.join(sim_dir,'simmat_mthrh045_{}_K{}_random.pickle'.format(category,vc_num))
	if not os.path.exists(savename):
		r_set = [None for nn in range(N)]
		for ii,data in enumerate(data_loader):
			input, demo_img, img_path, true_pad = data
			if imgs_par_cat[cat_idx]<N:
				with torch.no_grad():
					layer_feature = extractor(input.cuda(device_ids[0]))[0].detach().cpu().numpy()
				iheight,iwidth = layer_feature.shape[1:3]
				lff = layer_feature.reshape(layer_feature.shape[0],-1).T
				lff_norm = lff / (np.sqrt(np.sum(lff ** 2, 1) + 1e-10).reshape(-1, 1)) + 1e-10
				r_set[ii] = cdist(lff_norm, centers, 'cosine').reshape(iheight,iwidth,-1)
				imgs_par_cat[cat_idx]+=1

		print('Determine best threshold for binarization - {} ...'.format(category))
		nthresh=20
		magic_thhs=range(nthresh)
		coverage = np.zeros(nthresh)
		act_per_pix = np.zeros(nthresh)
		layer_feature_b = [None for nn in range(100)]
		magic_thhs = np.asarray([x*1/nthresh for x in range(nthresh)])
		for idx,magic_thh in en