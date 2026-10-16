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

X_og, X_test, y_og, y_test = load_and_split("tasic", test_size=0.2, seed=0, labels=True)
X, X_val, y, y_val = train_test_split(X_og, y_og, test_size = 0.2, random_state = 0)

seed_list = [0]
scdeed_best = {
    "tsne": {0: 2729},
    "umap": {0: 2000}
}
embedr_best = {
    "tsne": {0: 2729,  },
    "umap": {0: 2000, }
}
pcs_best = {
    "tsne": {0: 5, },
}
medal_best = {
    "tsne": {0: 53,},
    "umap": {0: 9,  }
}

teacher_embed = {}
for s in seed_list:
    teacher_embed[f'scdeed'] = np.load(Path(PATH_PREFIX) / f"MEDAL/comparisons/data/tasic_train_tsne_{scdeed_best['tsne'][s]}_0_train_pc4.npy")
    teacher_embed[f'medal'] = np.load(Path(PATH_PREFIX) / f"drd_data/embeddings2/tasic_tsne_{medal_best['tsne'][s]}_{s}_train.npy")
    teacher_embed[f'embedr'] = np.load(Path(PATH_PREFIX) / f"drd_data/embeddings/tasic_tsne_{embedr_best['tsne'][s]}_{s}_train_embedr.npy").squeeze(0)
    teacher_embed[f'pcs'] = np.load(Path(PATH_PREFIX) / f"drd_data/embeddings/tasic_tsne_{pcs_best['tsne'][s]}_{s}_train_pcs.npy")

by_k_tsne = pd.DataFrame([])
for s in seed_list:
    print("Eval MEDAL")
    tw_df = evaluate_embedding(
        X_high = X_og,
        X_low = get_teacher_embeddings("tsne", X_og, perplexity = medal_best["tsne"][s], learning_rate="auto", random_state=s),
        k_values = range(10, 301, 20)
    )
    tw_df.update({"method": f'MEDAL_SEED{s} ({medal_best["tsne"][s]})'})
    by_k_tsne = pd.concat([by_k_tsne, pd.DataFrame(tw_df)])
    
    print("Eval EMBEDR")
    tw_df = evaluate_embedding(
        X_high = X_og,
        X_low = teacher_embed[f"embedr"],
        k_values = range(10, 301, 20)
    )
    tw_df.update({"method": f'EMBEDR_SEED{s} ({embedr_best["tsne"][s]})'})
    by_k_tsne = pd.concat([by_k_tsne, pd.DataFrame(tw_df)])
    
    print("Eval PCS")
    tw_df = evaluate_embedding(
        X_high = X_og,
        X_low = teacher_embed[f"pcs"],
        k_values = range(10, 301, 20)
    )
    tw_df.update({"method": f'PCS_SEED{s} ({pcs_best["tsne"][s]})'})
    by_k_tsne = pd.concat([by_k_tsne, pd.DataFrame(tw_df)])
    
    print("Eval SCDEED")
    tw_df = evaluate_embedding(
        X_high = X_og,
        X_low = teacher_embed[f"scdeed"],
        k_values = range(10, 301, 20)
    )
    tw_df.update({"method": f'SCDEED_SEED{s} ({scdeed_best["tsne"][s]})'})
    by_k_tsne = pd.concat([by_k_tsne, pd.DataFrame(tw_df)])
    
by_k_tsne.to_csv('tasic_by_k_tsne_all_methods.csv')


# UMAP

teacher_embed = {}
teacher_embed[f'scdeed'] = np.load(Path(PATH_PREFIX) / f"MEDAL/comparisons/data/tasic_train_umap_2000_0.1_0_train_pc4.npy")
teacher_embed[f'embedr'] = np.load(Path(PATH_PREFIX) / f"drd_data/embeddings/tasic_umap_2000_0.1_0_train_embedr.npy").squeeze(0)

by_k_umap = pd.DataFrame([])
for s in seed_list:
    print("Eval MEDAL")
    tw_df = evaluate_embedding(
        X_high = X_og,
        X_low = get_teacher_embeddings("umap", X_og, n_neighbors = medal_best["umap"][s], min_dist = 0.1, random_state=s),
        k_values = range(10, 301, 20)
    )
    tw_df.update({"method": f'MEDAL (5)'})
    by_k_umap = pd.concat([by_k_umap, pd.DataFrame(tw_df)])
    
    print("Eval EMBEDR")
    tw_df = evaluate_embedding(
        X_high = X_og,
        X_low = teacher_embed[f"embedr"],
        k_values = range(10, 301, 20)
    )
    tw_df.update({"method": f'EMBEDR (2000)'})
    by_k_umap = pd.concat([by_k_umap, pd.DataFrame(tw_df)])
    
    print("Eval SCDEED")
    tw_df = evaluate_embedding(
        X_high = X_og,
        X_low = teacher_embed[f"scdeed"],
        k_values = range(10, 301, 20)
    )
    tw_df.update({"method": f'SCDEED (2000)'})
    by_k_umap = pd.concat([by_k_umap, pd.DataFrame(tw_df)])
    
by_k_umap.to_csv('tasic_by_k_umap_all_methods.csv')
