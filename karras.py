import torch
from utils import add_dims

def sigmas_karras(steps, sigma_min=2e-3, sigma_max=80., rho=7.):
    i = torch.linspace(0, 1, steps)
    sigma_max = sigma_max ** (1/rho)
    sigma_min = sigma_min ** (1/rho)
    sigmas = (sigma_max + i*(sigma_min-sigma_max))**rho
    return torch.cat([sigmas, torch.tensor([0])])


def karras_scaler(sigma, sigma_data, dims=None, device='cuda'):
    total_var = sigma**2 + sigma_data**2
    c_skip = sigma_data**2 / total_var
    c_out = sigma * sigma_data / total_var.sqrt()
    c_in = 1 / total_var.sqrt()
    if dims is not None:
        c_in, c_out, c_skip = add_dims(c_in, dims), add_dims(c_out, dims), add_dims(c_skip, dims)
    return c_in.to(device), c_out.to(device), c_skip.to(device)


def noisify_karras(xb, sigma_data=0., device='cuda'):
    device = xb.device
    sigma = (torch.randn([len(xb)])*1.2-1.2).exp().to(device).reshape(-1,1,1,1)
    noise = torch.randn_like(xb).to(device)
    c_in, c_out, c_skip = karras_scaler(sigma, sigma_data)
    xb_noised = xb + sigma*noise
    target = (1/c_out) * (xb - c_skip*xb_noised)
    return (c_in*xb_noised, sigma.squeeze()), target