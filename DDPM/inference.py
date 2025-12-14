import torch
from tqdm import tqdm

class DDPMInference:
    def __init__(self,loader,device="mps"):
        self.loader=loader
        self.device=device
        self.unet=self.loader.unet
        self.scheduler=self.loader.scheduler

    def run_inference(self,num_steps=50):
        self.scheduler.set_timesteps(num_steps,device=self.device)
        latents = torch.randn(
            (1, self.unet.config.in_channels, 64, 64),
            device=self.device,
            dtype=self.unet.dtype,
        )
        intermediates = []

        with torch.no_grad():
            for i, t in enumerate(tqdm(self.scheduler.timesteps)):
                noise_pred = self.unet(latents, t).sample

                intermediates.append({
                    "step": i,
                    "timestep": int(t),
                    "latent": latents.detach().cpu(),
                    "noise_pred": noise_pred.detach().cpu(),
                })

                latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        return intermediates, latents


