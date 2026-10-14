from pathlib import Path
import json, random, time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from captum.attr import IntegratedGradients
from medal.model import MEDAL
from medal.io import load_model
from medal.normalizer import GlobalEmbeddingNormalizer

OUT = Path(__file__).resolve().parent
SOURCE = Path('/Users/clairehe/Documents/GitHub/MEDAL_FI/encoder_FI/experiments/arch_search')

torch.set_num_threads(4)
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

cfg = json.loads((SOURCE/'best_config.json').read_text())['best_config']
X = np.load(SOURCE/'X_train.npy').astype('float32')
V = np.load(SOURCE/'X_val.npy').astype('float32')
raw = np.load(SOURCE/'embeddings/arch_search_teacher.npy')
norm = GlobalEmbeddingNormalizer.load(SOURCE/'embeddings/arch_search_teacher.norm.pkl')
Z = norm.transform(raw)
assert len(X)==len(Z) and np.isfinite(X).all() and np.isfinite(Z).all()
print(f'Training on {X.shape}, validation {V.shape}; original pixel scale retained.', flush=True)

kwargs = {k : cfg[k] for k in ['hidden_dims', 'activation', 'use_batchnorm', 'dropout_rate',
                               'lr', 'lambda_d', 'batch_size', 'warmup', 'eta_min',
                               'adamw_weight_decay', 'bottleneck_activation', 'final_activation']}
student = MEDAL(input_dim=64, latent_dim=2,epochs=cfg['max_epochs'],factor=cfg['t_factor'], 
                patience=cfg['t_patience'],device='cpu',**kwargs)

# Preserve MEDAL reporting while reducing output to meaningful progress updates.
import contextlib, io, sys
progress_stdout = sys.stdout

class Progress(io.TextIOBase):
 def __init__(self): 
  self.n = 0

 def write(self,s):
  if s.startswith('{'):
   self.n += 1
   if self.n%100 == 0: 
    print(f'Report {self.n}/1000: {s}', flush=True, file=progress_stdout)
  elif s.strip(): 
   print(s, flush=True, file=progress_stdout)
  return len(s)
 
with contextlib.redirect_stdout(Progress()):
 student.fit(X, Z, verbose=False, print_tag=True,
             save_dir=OUT, prefix='baby_mnist')

ckpt = OUT/'baby_mnist_ckpts/final.pt'
model = load_model(ckpt, 64, tuple(cfg['hidden_dims']), latent_dim = 2, activation = cfg['activation'],
                   use_batchnorm = cfg['use_batchnorm'], dropout_rate = cfg['dropout_rate'])
with torch.no_grad():
 recon, z = model(torch.from_numpy(X))
 vr, vz = model(torch.from_numpy(V))
 metrics={'train_distill_mse': float(((z-torch.from_numpy(Z))**2).mean()),
          'train_recon_mse': float(((recon-torch.from_numpy(X))**2).mean()),
          'validation_recon_mse': float(((vr-torch.from_numpy(V))**2).mean()),
          'validation_mean_baseline_mse': float(np.mean((V-X.mean(0))**2))}
assert all(np.isfinite(v) for v in metrics.values())
print('Metrics:', metrics, flush=True)

# Attribute both normalized latent coordinates on held-out input rows.
ig=IntegratedGradients(model.encoder)
attrs=[]
deltas=[]
for j in range(2):
 aa = []
 dd = []
 for start in range(0, len(V), 32):
  t = torch.from_numpy(V[start:start+32])
  a, d = ig.attribute(t, baselines = torch.zeros_like(t), target=j,
                      n_steps=100, internal_batch_size=128, 
                      return_convergence_delta=True)
  aa.append(a.detach().numpy())
  dd.append(d.detach().numpy())

 attrs.append(np.concatenate(aa))
 deltas.append(np.concatenate(dd))
attr=np.stack(attrs,axis=1)
delta=np.stack(deltas,axis=1)

assert np.isfinite(attr).all()

digits=load_digits()
lookup={tuple(row):int(label) for row,label in zip(digits.data,digits.target)}
labels=np.array([lookup[tuple(row)] for row in V])
importance=np.abs(attr).sum(axis=1)

np.savez_compressed(OUT/'encoder_ig.npz',
                    X=V,
                    labels=labels,
                    attributions=attr,
                    convergence_delta=delta,
                    global_importance=importance.mean(0),
                    latent=vz.numpy())
metrics['ig_mean_abs_completeness_error']=float(np.abs(delta).mean())
metrics['ig_max_abs_completeness_error']=float(np.abs(delta).max())
(OUT/'metrics.json').write_text(json.dumps(metrics,indent=2))
(OUT/'config.json').write_text(json.dumps(dict(cfg,
                                               input_dim=64,
                                               latent_dim=2,
                                               seed=0, 
                                               source=str(SOURCE)),
                                               indent=2))
np.savez(OUT/'teacher_normalization.npz', mean=norm.mean_, scale=norm.scale_)

fig, axes = plt.subplots(2, 10, figsize=(15,4), layout='constrained')
for k in range(10):
 sel=labels==k
 axes[0,k].imshow(V[sel].mean(0).reshape(8,8),cmap='gray');axes[0,k].set_title(str(k))
 axes[1,k].imshow(importance[sel].mean(0).reshape(8,8),cmap='magma',vmin=0)
 for ax in axes[:,k]:ax.axis('off')
fig.suptitle('Baby MNIST: mean validation image (top), encoder IG magnitude (bottom)\nZero baseline; |IG(z₁)| + |IG(z₂)|; each heatmap scaled independently')
fig.savefig(OUT/'encoder_ig.png',dpi=180);plt.close(fig)
print('FINISHED',OUT,flush=True)
