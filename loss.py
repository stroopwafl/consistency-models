import torch
from torch import tensor
import torch.nn.functional as F
from utils import get_device
import fastcore.all as fc
from functools import partial


class Hook:
    """
        Base hook class that initialises a Pytorch hook using the function
        passed as an argument.
    """
    def __init__(self, model, func): self.hook = model.register_forward_hook(partial(func, self))
    def remove(self): self.hook.remove()
    def __del__(self): self.remove()


def imgnet_normalise(img):
    device = img.device
    imagenet_mean = tensor([0.485, 0.456, 0.406], device=device)
    imagenet_std = tensor([0.229, 0.224, 0.225], device=device)
    return ((img - imagenet_mean[:,None,None])/imagenet_std[:,None,None])


def get_vgg_feats(hook, module, inp, out):
    if not hasattr(hook, 'vgg_feats'): hook.vgg_feats = out
    hook.vgg_feats = out
    

def get_features(img, feat_model, target_layers:tuple=(18,25), device='cuda:0'):
    x = imgnet_normalise(img.to(device))
    hooks = [Hook(l, get_vgg_feats) for i,l in enumerate(feat_model) if i in target_layers]
    feat_model[:max(target_layers)+1](x)
    feats = []
    for h in hooks:
        feats.append(h.vgg_feats)
        h.remove()
    return feats


class LPIPS:
    def __init__(self, feat_model, target_layers:tuple=(3,8,15,22,29), device='cuda:0'): fc.store_attr()
    def __call__(self, x, y):
        x_feats = get_features((F.interpolate(x, size=224, mode='bilinear'))+0.5, self.feat_model, self.target_layers, device=self.device) 
        y_feats = get_features((F.interpolate(y, size=224, mode='bilinear'))+0.5, self.feat_model, self.target_layers, device=self.device)
        return sum((f1-f2).pow(2).mean() for f1,f2 in zip(x_feats, y_feats))