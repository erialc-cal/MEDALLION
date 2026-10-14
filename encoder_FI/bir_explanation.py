import numpy as np
from pathlib import Path
import sys
sys.path.append('../')
from benchmarking.bir import generate_samples_SMOTE, query_blackbox_tsne, explain_samples
from sklearn.linear_model import ElasticNet
from sklearn.datasets import load_digits
import tqdm 

folder = Path("/Users/clairehe/Documents/GitHub/MEDAL_FI/encoder_FI/experiments/arch_search")
X_train = np.load(folder / "X_train.npy")
X_val = np.load(folder / 'X_val.npy')
Z = np.load( folder / "embeddings/arch_search_teacher.npy")
digits = load_digits()
label_lookup = {tuple(image): label for image, label in zip(digits.data, digits.target)}
y_train = np.array([label_lookup[tuple(image)] for image in X_train])

BIR_attr = np.zeros((X_train.shape[0], 2, 64))
for i in tqdm.tqdm(range(X_train.shape[0])):
    # Create explanation for each instance
    x_samples = np.asarray(generate_samples_SMOTE(selected_id=i, X=X_train, k_nearbors=10, n_samples=10)).reshape(-1, X_train.shape[1])
    y_samples = query_blackbox_tsne(X=X_train, Y=Z, query_idx=i, query_points=x_samples,
    tsne_hyper_params={"n_components": 2, "perplexity": 30, "random_state": 42, "method": "exact", 'max_iter':300, 'verbose':0})
    W, r2, rotation = explain_samples(x_samples, y_samples, linear_model=ElasticNet())
    # W.shape (2, 64)
    BIR_attr[i, : ,:] = W

import pickle
with open('experiments/arch_search/comp_files/bir_expl.pkl', 'wb') as file:
    pickle.dump(BIR_attr, file) # (N_tr, Z_dim, feat_dim)