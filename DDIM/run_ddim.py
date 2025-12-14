
import matplotlib.pyplot as plt

from .model_loader import DDIM_Loader
from .inference import DDIMInference
from ..extract_steps import Save_Steps  

loader = DDIM_Loader(device="mps")
infer = DDIMInference(loader, device="mps")


intermediates, final_image = infer.run_inference(num_steps=1000, eta=0.0)


saver = Save_Steps(output_dir="project/data/ddim_steps")
saver.save_steps(intermediates)


img = final_image.detach().cpu()
img = (img + 1) / 2       
img = img.clamp(0, 1)
img_np = img.squeeze(0).permute(1, 2, 0).numpy()

plt.figure(figsize=(4, 4))
plt.imshow(img_np)
plt.axis("off")
plt.show()
