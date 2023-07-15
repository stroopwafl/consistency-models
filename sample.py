import torch
from utils import add_dims
from karras import sigmas_karras, karras_scaler
from fastprogress import progress_bar


@torch.no_grad()
def karras_sample(sample_func, size, model, steps=100, device='cuda', sigma_max=80., sigma_data=0.):
    x = torch.randn(size).to(device) * sigma_max
    sigma = sigmas_karras(steps).to(device)
    for t in progress_bar(range(steps)):
        sig_t, sig_t1 = sigma[t], sigma[t+1]
        x = sample_func(x, sig_t, sig_t1, model, sigma_data)
    return x


@torch.no_grad()
def consistency_sample(size, model, steps=100, device='cuda', sigma_min=2e-3, sigma_max=80., sigma_data=0.):
    x = torch.randn(size).to(device) * sigma_max
    sigma = sigmas_karras(steps).to(device)
    x = model((x, torch.tensor(sigma_max)), sigma_data=sigma_data)
    if steps <= 1: return x
    for t in progress_bar(range(steps-1)):
        z = torch.randn_like(x)
        xt = x + (sigma[t]**2 - sigma_min**2).sqrt()*z
        x = model((xt, (sigma[t]**2 - sigma_min**2).sqrt()), sigma_data=sigma_data)
    return x


@torch.no_grad()
def euler(x, t, next_t, model):
    d1 = (x - model((x,t))) / add_dims(t, x)
    return x + d1 * add_dims(next_t - t, x)


# @torch.no_grad()
# def heun(x, t, next_t, model):
#     d1 = (x - model((x,t))) / add_dims(t, x)
#     x1 = x + d1 * add_dims(next_t - t, x)
#     if next_t == 0: return x1
#     d2 = (x1 - model((x,t))) / add_dims(next_t, x)
#     d_avg = (d1+d2)/2
#     return x + d_avg * add_dims(next_t - t, x)

@torch.no_grad()
def heun(x, sig, sig2, model, sigma_data=0.):
    denoised = denoise_fn(x, sig, model, sigma_data)
    d = (x-denoised)/sig
    x1 = x + d*(sig2-sig)
    if sig2 == 0: return x1
    x1_denoised = denoise_fn(x1, sig2, model, sigma_data)
    d1 = (x1-x1_denoised)/sig2
    d_average = (d1+d)/2
    return x + d_average*(sig2-sig)


def denoise_fn(x, t, model, sigma_data, scaler=karras_scaler):
    c_in, c_out, c_skip = scaler(t, sigma_data, dims=x, device=x.device)
    return model((c_in*x, t))*c_out + c_skip*x