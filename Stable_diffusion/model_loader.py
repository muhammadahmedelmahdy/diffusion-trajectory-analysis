## Imports

import torch
from diffusers import StableDiffusionPipeline


class Stable_Diffusion_Loader:
    def __init__(self,model="runwayml/stable-diffusion-v1-5",device="mps"):
        self.model=model
        self.device=device
        self.load_model()

    def load_model(self):
        self.pipeline=StableDiffusionPipeline.from_pretrained(self.model,torch_dtype=torch.float32).to(self.device)
        self.pipeline.enable_attention_slicing()
        self.pipeline.vae.enable_tiling()
        self.unet=self.pipeline.unet
        self.vae=self.pipeline.vae
        self.text_encoder=self.pipeline.text_encoder
        self.tokenizer=self.pipeline.tokenizer
        self.scheduler=self.pipeline.scheduler

    def encode_prompt(self,prompt):
        tokens = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length
        ).to(self.device)

        return self.text_encoder(tokens.input_ids)[0]
    
    def decode_latents(self,latents):
        latents = latents / self.vae.config.scaling_factor
        with torch.no_grad():
            generated_image=self.vae.decode(latents).sample
        
        return generated_image
    
    def get_components(self):
        return {
            "unet": self.unet,
            "vae": self.vae,
            "scheduler": self.scheduler,
            "text_encoder": self.text_encoder,
            "tokenizer": self.tokenizer,
        }




        

