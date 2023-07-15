import torch, math, copy
from utils import add_dims
import fastcore.all as fc
from karras import noisify_karras, sigmas_karras, karras_scaler

from miniai.learner import *
from miniai.datasets import *
from miniai.conv import *
from miniai.activations import *
from miniai.core import *
from miniai.initialisation import *
from miniai.accel import *
from miniai.ddpm import *
from miniai.resnet import *
from miniai.aug import *
from miniai.fid import *
from miniai.ddim import *


def consistency_scaler(sigma, sigma_data, sigma_min=2e-3, dims=None, device='cuda'):
    c_in = 1 / (sigma**2 + sigma_data**2).sqrt()
    c_out = (sigma_data*(sigma-sigma_min)) / (sigma_data**2 + sigma**2).sqrt()
    c_skip = sigma_data**2 / ((sigma - sigma_min)**2 + sigma_data**2)
    if dims is not None:
        c_in, c_out, c_skip = add_dims(c_in, dims), add_dims(c_out, dims), add_dims(c_skip, dims)
    return c_in.to(device), c_out.to(device), c_skip.to(device)


def n_sched(step, n_steps, s0=2, s1=150):
    step = torch.tensor(step)
    n = torch.sqrt((step * ((s1+1)**2 - s0**2) / n_steps) + s0**2) - 1
    return (torch.ceil(n) + 1).to(torch.long)


def mu_sched(n, s0=2, mu0=0.95):
    return (s0 * math.log(mu0) / torch.tensor(n)).exp()


def noisify_consistency(xb, n, sigma_min=2e-3, sigma_max=80., device='cuda'):
    noise = torch.randn_like(xb, device=device)
    sigma = sigmas_karras(n, sigma_min, sigma_max).to(device)
    t = torch.randint(0, n-1, (len(xb), ), device=device)
    sig_t, sig_t1 = sigma[t], sigma[t+1]
    xt = xb + add_dims(sig_t, xb)*noise
    return xt, sig_t, sig_t1


def denoise_fn(x, t, model, sigma_data, scaler=karras_scaler):
    c_in, c_out, c_skip = scaler(t, sigma_data, dims=x, device=x.device)
    return model((c_in*x, t))*c_out + c_skip*x


@torch.no_grad()
def euler_solve(xb, xt, t, next_t, model=None):
    denoiser = xb if model is None else denoise_fn(xt, t, model)
    d1 = (xt - denoiser) / add_dims(t, xt)
    return xt + d1 * add_dims(next_t - t, xt)


@torch.no_grad()
def heun_solve(xb, xt, t, next_t, sigma_data, model=None):
    denoiser = xb if model is None else denoise_fn(xt, t, model, sigma_data)
    d1 = (xt - denoiser) / add_dims(t, xt)
    x1 = xt + d1 * add_dims(next_t - t, xt)
    denoiser = xb if model is None else denoise_fn(x1, next_t, model, sigma_data)
    d2 = (x1 - denoiser) / add_dims(next_t, xt)
    d_avg = (d1+d2)/2
    return xt + d_avg * add_dims(next_t - t, xt)


class ConsistencyCB(MixedPrecisionCB):
    def __init__(self, teacher_model=None, training_mode='training', sigma_min=2e-3, sigma_max=80., sigma_data=0.33, ema_decay=0.9999, mu0=0.9, use_ema=True, distil_n=18, device='cuda:0'):
        super().__init__()
        fc.store_attr()
        if self.teacher_model is None and self.training_mode == 'distillation': raise Exception(
            "To do distilllation, provide a pretrained model. Otherwise, use training_mode='training'")
        self.ode_solve = heun_solve if self.training_mode == 'distillation' else euler_solve
        
    def before_fit(self):
        self.scaler = torch.cuda.amp.GradScaler()
        if self.training_mode == 'distillation':
            self.teacher_model = self.teacher_model.to(self.device).requires_grad_(False)
            self.learn.model.load_state_dict(self.teacher_model.state_dict())
        elif self.training_mode == 'training':
            self.steps = self.learn.n_epochs*len(self.learn.dls.train)
        if self.use_ema:
            self.online_ema = copy.deepcopy(self.learn.model).to(self.device).requires_grad_(False)
        self.target_model = copy.deepcopy(self.learn.model).to(self.device).requires_grad_(False)
        
    def predict(self):
        xb, device = self.learn.xb, self.learn.xb.device
        self.it = (self.learn.epoch)*len(self.learn.dls.train)+self.learn.batch_idx
        n = n_sched(self.it, self.steps) if self.training_mode == 'training' else self.distil_n
        xt, t, next_t = noisify_consistency(xb, n, self.sigma_min, self.sigma_max, device=self.device)
        self.learn.preds = self.learn.model((xt, t), self.sigma_data)
        with torch.no_grad():
            xt1 = self.ode_solve(xb, xt, t, next_t, self.sigma_data, model=self.teacher_model).detach()
            self.learn.batch = self.learn.xb, self.target_model((xt1, next_t)).detach()
    
    def step(self):
        self.scaler.step(self.learn.opt)
        self.scaler.update()
        mu = self.mu0
        if self.training_mode == 'training': mu = mu_sched(self.it)
        if self.use_ema:
            for p, ema_p in zip(self.learn.model.parameters(), self.online_ema.parameters()):
                ema_p = self.ema_decay*ema_p + (1-self.ema_decay)*p
        for online_p, target_p in zip(
            self.online_ema.parameters() if self.use_ema else self.learn.model.parameters(),
            self.target_model.parameters()
        ):
            target_p = mu*target_p + (1-mu)*online_p