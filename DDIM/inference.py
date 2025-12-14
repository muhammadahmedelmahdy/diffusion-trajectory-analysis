
import torch
from tqdm import tqdm


class DDIMInference:
    def __init__(self,loader,device="mps"):
        self.loader=loader
        self.device=device
        self.unet=loader.unet
        self.scheduler=loader.scheduler

    
    def run_inference(self,num_steps=50,eta=0.0):
        self.scheduler.set_timesteps(num_steps,device=self.device)
        # Initial noise: [1, C, H, W]
        image = torch.randn(
            (1, self.unet.config.in_channels, 64, 64),
            device=self.device,
            dtype=self.unet.dtype,
        )

        intermediates = []
        with torch.no_grad():
            for i, t in enumerate(tqdm(self.scheduler.timesteps)):
                
                noise_pred = self.unet(image, t).sample

                
                intermediates.append({
                    "step": i,
                    "timestep": int(t),
                    "latent": image.detach().cpu(),        
                    "noise_pred": noise_pred.detach().cpu()
                })

                
                step_output = self.scheduler.step(
                    model_output=noise_pred,
                    timestep=t,
                    sample=image,
                    eta=eta
                )
                image = step_output.prev_sample

        return intermediates, image




        

