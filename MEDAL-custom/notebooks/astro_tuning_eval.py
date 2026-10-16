from structure_preservation_metrics import evaluate_embedding
from eval.data import load_and_split
from medal.teacher import get_teacher_embeddings
from sklearn.model_selection import train_test_split
import numpy as np, pandas as pd
from pathlib import Path
import torch

# Set MEDAL_DATA_DIR to the parent directory of your data (drd_data/ lives here)
import os
PATH_PREFIX = os.environ.get('MEDAL_DATA_DIR', os.path.expanduser('~'))

X_og, X_test, y_og, y_test = load_and_split("astro", test_size=0.2, seed=0, labels=True)
X, X_val, y, y_val = train_test_split(X_og, y_og, test_size = 0.2, random_state = 0)


teacher_embed = {}
teacher_embed[f'scdeed'] = np.load(Path(PATH_PREFIX) / f"MEDAL/comparisons/data/astro_train_tsne_499_0_train_pc5.npy")
teacher_embed[f'medal'] = np.load(Path(PATH_PREFIX) / f"drd_data/embeddings2/astro_tsne_6_0_train.npy")
teacher_embed[f'embedr'] = np.load(Path(PATH_PREFIX) / f"drd_data/embeddings/astro_tsne_499_0_train_embedr.npy").squeeze(0)
teacher_embed[f'pcs'] = np.load(Path(PATH_PREFIX) / f"drd_data/embeddings/astro_tsne_3_0_train_pcs.npy")

by_k_tsne = pd.DataFrame([])

print("Eval MEDAL")
tw_df = evaluate_embedding(
    X_high = X_og,
    X_low = get_teacher_embeddings("tsne", X_og, perplexity =6, learning_rate="auto", random_state=0),
    k_values = range(5, 21)
)
tw_df.update({"method": f'MEDAL (6)'})
by_k_tsne = pd.concat([by_k_tsne, pd.DataFrame(tw_df)])

print("Eval EMBEDR")
tw_df = evaluate_embedding(
    X_high = X_og,
    X_low = teacher_embed[f"embedr"],
    k_values = range(5, 21)
)
tw_df.update({"method": f'EMBEDR (499)'})
by_k_tsne = pd.concat([by_k_tsne, pd.DataFrame(tw_df)])

print("Eval PCS")
tw_df = evaluate_embedding(
    X_high = X_og,
    X_low = teacher_embed[f"pcs"],
    k_values = range(5, 21)
)
tw_df.update({"method": f'PCS (3)'})
by_k_tsne = pd.concat([by_k_tsne, pd.DataFrame(tw_df)])

print("Eval SCDEED")
tw_df = evaluate_embedding(
    X_high = X_og,
    X_low = teacher_embed[f"scdeed"],
    k_values = range(5, 21)
)
tw_df.update({"method": f'SCDEED (499)'})
by_k_tsne = pd.concat([by_k_tsne, pd.DataFrame(tw_df)])
    
by_k_tsne.to_csv('astro_by_k_tsne_all_methods.csv')

# UMAP

teacher_embed = {}
teacher_embed[f'scdeed'] = np.load(Path(PATH_PREFIX) / f"MEDAL/comparisons/data/mnist_train_umap_134_0.1_0_train_pc6.npy")
teacher_embed[f'medal'] = np.load(Path(PATH_PREFIX) / f"drd_data/embeddings2/mnist_umap_9_0.1_0_train.npy")
teacher_embed[f'embedr'] = np.load(Path(PATH_PREFIX) / f"drd_data/embeddings/mnist_umap_499_0.1_0_train_embedr.npy").squeeze(0)
        
by_k_umap = pd.DataFrame([])
for s in [0]:
    print("Eval MEDAL")
    tw_df = evaluate_embedding(
        X_high = X_og,
        X_low = get_teacher_embeddings("umap", X_og, n_neighbors = 9, min_dist = 0.1, random_state=s),
        k_values = range(5, 21)
    )
    tw_df.update({"method": f'MEDAL (9)'})
    by_k_umap = pd.concat([by_k_umap, pd.DataFrame(tw_df)])
    
    print("Eval EMBEDR")
    tw_df = evaluate_embedding(
        X_high = X_og,
        X_low = teacher_embed[f"embedr"],
        k_values = range(5, 21)
    )
    tw_df.update({"method": f'EMBEDR (499)'})
    by_k_umap = pd.concat([by_k_umap, pd.DataFrame(tw_df)])
    
    print("Eval SCDEED")
    tw_df = evaluate_embedding(
        X_high = X_og,
        X_low = teacher_embed[f"scdeed"],
        k_values = range(5, 21)
    )
    tw_df.update({"method": f'SCDEED (134)'})
    by_k_umap = pd.concat([by_k_umap, pd.DataFrame(tw_df)])
    
by_k_umap.to_csv('astro_by_k_umap_all_methods.csv')

