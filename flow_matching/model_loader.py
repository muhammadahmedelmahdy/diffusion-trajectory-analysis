import torch
from diffusers import StableDiffusion3Pipeline


class SD3FlowMatchingLoader:
    def __init__(self, device="mps", dtype=torch.float16):
        self.device = device
        self.dtype = dtype
        self.load_model()

    def load_model(self):
        # Load pipeline
        self.pipe = StableDiffusion3Pipeline.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers",
            torch_dtype=self.dtype,
        ).to(self.device)

        
        

        self.pipe.transformer.to(dtype=self.dtype)
        self.pipe.vae.to(dtype=self.dtype)
        self.pipe.text_encoder.to(dtype=self.dtype)

        
        self.transformer = self.pipe.transformer
        self.scheduler = self.pipe.scheduler
        self.vae = self.pipe.vae
        self.text_encoder = self.pipe.text_encoder
        self.tokenizer = self.pipe.tokenizer

    def get_components(self):
        return {
            "pipe": self.pipe,
            "transformer": self.transformer,
            "scheduler": self.scheduler,
            "vae": self.vae,
            "text_encoder": self.text_encoder,
            "tokenizer": self.tokenizer,
        }
