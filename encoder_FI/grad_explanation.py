# Gradient explanation is expensive to compute 

import medal
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pickle
import shap
import torch.nn as nn
import torch
import json
from pathlib import Path

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from functools import partial
import numpy as np
from sklearn.datasets import load_digits
from pathlib import Path

folder = Path("/Users/clairehe/Documents/GitHub/MEDAL_FI/encoder_FI/experiments/arch_search")
X_train = np.load(folder / "X_train.npy")
X_val = np.load(folder / 'X_val.npy')
Z = np.load( folder / "embeddings/arch_search_teacher.npy")
digits = load_digits()
label_lookup = {tuple(image): label for image, label in zip(digits.data, digits.target)}
y_train = np.array([label_lookup[tuple(image)] for image in X_train])

import sys
sys.path.append('../')
from benchmarking._t_sne import TSNE
from benchmarking.grad_exp import *
tsne = TSNE(n_components=2, perplexity=30, random_state=42, method='exact')
feat=tsne.fit_transform(X_train)

import tqdm
grad_expl=[]
for idx, i in tqdm.tqdm(enumerate(y_train)):
    direct_2order=calc_direct_2order(tsne, feat, idx)
    cross_2order_tot=cross_2order_total(tsne, X_train, feat, idx)
    grad_x=grad_wtr_x(direct_2order, cross_2order_tot)
    l2 = np.sqrt((grad_x[0])**2 + (grad_x[1]**2))
    grad_expl.append(l2)


import pickle
with open('experiments/arch_search/comp_files/grad_exp.pkl', 'wb') as file:
    pickle.dump(grad_expl, file)