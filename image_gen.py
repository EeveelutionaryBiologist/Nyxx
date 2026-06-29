
import torch
from diffusers import Flux2KleinPipeline, StableDiffusionPipeline


IMAGE_MODEL = "FLUX2"

# ======================================================================
# Set image gen device
# ======================================================================
def to_device(pipe):
    if torch.cuda.is_available():
        return pipe.to("cuda") # For Nvidia GPUs
    elif torch.backends.mps.is_available():
        return pipe.to("mps")  # For Apple Silicon (M1/M2/M3/M4)
    else:
        return pipe.to("cpu")  # Fallback 

"""
dtype = torch.bfloat16

pipe = Flux2KleinPipeline.from_pretrained("black-forest-labs/FLUX.2-klein-9B", torch_dtype=dtype)
pipe.enable_model_cpu_offload()  # save some VRAM by offloading the model to CPU

pipe = to_device(pipe)

prompt = "An axolotl holding a sign with the CachyOS logo"
image = pipe(
    prompt=prompt,
    height=1024,
    width=1024,
    guidance_scale=1.0,
    num_inference_steps=4,
    generator=torch.Generator(device=device).manual_seed(0)
).images[0]
image.save("flux-klein.png")
"""

model_id = "gsdf/Counterfeit-V2.5" 

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = to_device(pipe)

image = pipe("Girl sitting on a bench, frilly dress, bare feet, beautiful, sun, grass").images[0]
image.save("counterfeit.png")
