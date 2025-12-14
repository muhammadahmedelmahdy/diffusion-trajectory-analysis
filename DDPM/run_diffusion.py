from .model_loader import DDPM_Loader
from .inference import *
import matplotlib.pyplot as plt
import numpy as np
from ..extract_steps import *

stable_diffusion=DDPM_Loader(device="mps")
inference=DDPMInference(stable_diffusion)


intermediates, image = inference.run_inference( num_steps=1000)
saver = Save_Steps("/Users/muhammadelmahdi/Prof_Ommer-Lab-Task/project/data/diffusion_steps")
saver.save_steps(intermediates)


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
