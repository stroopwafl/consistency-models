import torch


def get_device():
    """
        Returns the available device in the current environment as
        a string.
    """
    if torch.backends.mps.is_available(): device = 'mps' 
    if torch.cuda.is_available(): device = 'cuda'
    else: device = 'cpu'
    return device


def add_dims(x,y):
    dim = y.ndim - x.ndim
    return x[(...,) + (None,)*dim]