import torch
from tqdm import tqdm

class StableDiffusionInference:
    def __init__(self,loader,device="mps"):
        self.loader=loader
        self.device=device

        self.unet=self.loader.unet
        self.vae=self.loader.vae
        self.text_encoder=self.loader.text_encoder
        self.tokenizer=self.loader.tokenizer
        self.scheduler=self.loader.scheduler

    def run_inference(self,prompt,num_steps=100):
        text_embeds=self.loader.encode_prompt(prompt)
        self.scheduler.set_timesteps(num_steps)
        latents=torch.randn(
            (1, self.unet.in_channels, 64, 64),
            device=self.device,
            dtype=torch.float32
        )
        intermediates=[]
        with torch.no_grad():
            for i,t in enumerate(tqdm(self.scheduler.timesteps)):
                latent_model_input=self.scheduler.scale_model_input(latents,t)
                prediction_of_noise=self.unet(latent_model_input,t,text_embeds).sample
                intermediates.append({
                    "step": i,
                "timestep": t.item(),
                "latent": latents.detach().cpu(),
                "noise_pred": prediction_of_noise.detach().cpu()
                })

                latents=self.scheduler.step(prediction_of_noise,t,latents).prev_sample
            

        return intermediates,latents
    
    def decode_final_latent(self,final_latent):
        return self.loader.decode_latents(final_latent)

        

