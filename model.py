import torch
from diffusers import DiffusionPipeline
import os
from dotenv import load_dotenv
load_dotenv()

pipe = DiffusionPipeline.from_pretrained(
    "playgroundai/playground-v2.5-1024px-aesthetic",
    torch_dtype=torch.float16,
    variant="fp16",
    # use_auth_token=os.getenv("HF_TOKEN")
).to("mps")

prompt = "A cookie monster devouring a ‘Password’ input field, panic in UI. Style: Pixar-style 3D, vibrant, 4k, humorous lighting."
# num_inference_steps default 50, how many denoising steps
# guidance scale default 3, controls how closely the generated image adheres to the text prompt
image = pipe(prompt=prompt, num_inference_steps=50, guidance_scale=3).images[0]
image.save("cookie_monster3.png")