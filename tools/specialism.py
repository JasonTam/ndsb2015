# Based on HD-CNN: HIERARCHICAL DEEP CONVOLUTIONAL
# NEURAL NETWORK FOR IMAGE CLASSIFICATION
# by Yan. Jagadeesh, DeCoste, Di, Piramuthu
# http://arxiv.org/pdf/1410.0736v3.pdf

from collections import defaultdict
from itertools import takewhile
from pprint import pprint
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import AffinityPropagation as AP
import os
import pickle
import numpy as np

curdir, _ = os.path.split(__file__)
PRED_PATH = './naive_prediction.p'
f_path_pred = os.path.join(curdir, PRED_PATH)


def make_cluster_map(damping=0.98):
	test_labels, prediction = pickle.load(open(f_path_pred, 'rb'))
	prob_conf = np.zeros((121, 121))
	for l in range(121):
		inds = np.squeeze(np.array(np.where(test_labels == l)))
		class_conf = prediction[inds, :].mean(axis=0)
		prob_conf[l, :] = class_conf
	F = prob_conf
	D = (1-F)
	np.fill_diagonal(D, 0)
	D_p = 0.5*(D+D.T)


	clst = AP(damping=damping, # damping determines # of clusters
			  max_iter=500, 
			  convergence_iter=15, 
			  affinity='euclidean', 
			  verbose=False)
	clst.fit(D_p)
	print 'Number of cluster:', len(clst.cluster_centers_)
	membership = np.c_[range(121), clst.labels_]

	fine_to_coarse = dict(membership)
	coarse_to_fine = {l: [] for l in clst.labels_}
	for k, v in fine_to_coarse.items():
		coarse_to_fine[v].append(k)
		
	pickle.dump(coarse_to_fine, open(os.path.join(curdir, 'coarse_to_fine.p'), 'wb'))
	pickle.dump(fine_to_coarse, open(os.path.join(curdir, 'fine_to_coarse.p'), 'wb'))
    
try:
	coarse_to_fine = pickle.load(open(os.path.join(curdir, 'coarse_to_fine.p'), 'rb'))
	fine_to_coarse = pickle.load(open(os.path.join(curdir, 'fine_to_coarse.p'), 'rb'))
except IOError:
	make_cluster_map()












