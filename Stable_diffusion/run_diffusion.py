from .model_loader import Stable_Diffusion_Loader
from .inference import *
import matplotlib.pyplot as plt
import numpy as np
from ..extract_steps import *

stable_diffusion=Stable_Diffusion_Loader(device="mps")
inference=StableDiffusionInference(stable_diffusion)


intermediates, final_latent = inference.run_inference("Football", num_steps=999)
saver = Save_Steps("/Users/muhammadelmahdi/Prof_Ommer-Lab-Task/project/data/latent_diffusion_steps")
saver.save_steps(intermediates)

image = inference.decode_final_latent(final_latent)
image = image.detach().cpu()

# Plot the image
image = (image + 1) / 2
image = image.clamp(0, 1)
image_np = image.squeeze(0).permute(1, 2, 0).numpy()

# Display
plt.figure(figsize=(6, 6))
plt.imshow(image_np)
plt.axis("off")
plt.show()
