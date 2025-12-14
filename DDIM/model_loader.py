
import torch
from diffusers import DDIMPipeline

class DDIM_Loader:
    def __init__(self,model="google/ddpm-cifar10-32",device="mps"):
        self.model=model
        self.device=device
        self.load_model()

    def load_model(self):
        self.pipeline = DDIMPipeline.from_pretrained(
            self.model,
            torch_dtype=torch.float32,   
        )

        self.pipeline.to(self.device)

        self.unet=self.pipeline.unet
        self.scheduler=self.pipeline.scheduler

    def get_components(self):
        return {
            "unet": self.unet,
            "scheduler": self.scheduler,
        }


    



        
