from diffusers import UNet2DModel
from consistency import consistency_scaler
from utils import add_dims


class Unet(UNet2DModel):
    def forward(self, x): return super().forward(*x).sample


class ConsistencyModel(UNet2DModel):
    def forward(self, inp, sigma_data=0.):
        x, sigma = inp
        c_in, c_out, c_skip = consistency_scaler(sigma, sigma_data, dims=x, device=x.device)
        out = super().forward(c_in*x, sigma).sample
        return c_out*out + c_skip*x