from __future__ import annotations
import math, random, torch, matplotlib.pyplot as plt, numpy as np, matplotlib as mpl
from pathlib import Path
from operator import itemgetter
from itertools import zip_longest
from functools import partial
import fastcore.all as fc

from torch import tensor, nn, optim
import torch.nn.functional as F
from datasets import load_dataset
from tqdm.auto import tqdm
import torchvision.transforms.functional as TF
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, default_collate

from .learner import *
from .datasets import *
from .conv import *
from .activations import *
from .core import *
from .initialisation import *
from .accel import *
from .resnet import *
from .aug import *
from .ddpm import *

def get_acts(hook, model, inp, out):
    if not hasattr(hook, 'acts'): hook.acts = []
    hook.acts.append(to_cpu(out))

# %% ../nbs/10_fid.ipynb 27
from scipy import linalg

# %% ../nbs/10_fid.ipynb 28
def _calc_stats(features):
    features = features.squeeze()
    return features.mean(0).detach().cpu(), features.T.cov().detach().cpu()

def _calc_fid(m1, c1, m2, c2):
    csr = tensor(linalg.sqrtm(c1@c2, 256).real)
    return (((m1-m2)**2).sum() + c1.trace() + c2.trace() - 2*csr.trace()).item()

# %% ../nbs/10_fid.ipynb 34
class FID():
    def __init__(self, dls, class_model, layer_index): 
        fc.store_attr()
        self.learn = BaseLearner(dls, class_model, opt_func=fc.noop, loss_func=fc.noop, cbs=[DeviceCB()])
        self.hook = Hook(class_model[layer_index], get_acts)
        self.learn.fit(0.1, 1, train=False)
        self.ref_feats = self.hook.acts[0].float().cpu()
    def get_features(self, samples):
        hook2 = Hook(self.class_model[self.layer_index], get_acts)
        self.learn.dls = DataLoaders([],[(samples, tensor(1))])
        self.learn.fit(0.1, 1, train=False)
        return torch.cat(hook2.acts).float().cpu().squeeze()
    def fid(self, samples):
        ref_stats, sample_stats = _calc_stats(self.ref_feats), _calc_stats(self.get_features(samples))
        return _calc_fid(*ref_stats, *sample_stats)