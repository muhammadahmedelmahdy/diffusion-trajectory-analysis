from .model_loader import SD3FlowMatchingLoader
from .inference import SD3FlowMatchingInference


loader = SD3FlowMatchingLoader(device="mps")
infer = SD3FlowMatchingInference(loader, save_every=1)

intermediates, image = infer.run_inference(
    prompt="a snowy German village at sunrise, highly detailed",
    negative_prompt="low quality, distorted",
    num_steps=100,
    guidance_scale=3.0,
    capture_steps=True,
)


image.save("sd3_medium.png")
print("[INFO] Saved sd3_medium.png")

print(f"[INFO] Captured {len(intermediates)} intermediate steps")
