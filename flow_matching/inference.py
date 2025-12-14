import torch


class SD3FlowMatchingInference:
    def __init__(self, loader, save_every=1):
        self.pipe = loader.pipe
        self.save_every = save_every  # subsample steps

    def run_inference(
        self,
        prompt,
        negative_prompt=None,
        num_steps=1,
        guidance_scale=5.0,
        capture_steps=False,
    ):
        intermediates = []

        def save_step(pipeline, step, timestep, callback_kwargs):
            if step % self.save_every != 0:
                return callback_kwargs

            latents = callback_kwargs["latents"]

            intermediates.append({
                "step": step,
                "timestep": int(timestep),
                "latent": latents.detach().cpu(),
            })

            
            del latents
            torch.mps.empty_cache()

            return callback_kwargs

        with torch.no_grad():
            output = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                callback_on_step_end=save_step if capture_steps else None,
                callback_on_step_end_tensor_inputs=["latents"] if capture_steps else None,
            )

        final_image = output.images[0]
        return intermediates, final_image
