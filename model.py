import torch
from diffusers import DiffusionPipeline
from dotenv import load_dotenv

load_dotenv(verbose=True, override=True)

DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available(
) else "mps" if torch.backends.mps.is_available() else "cpu")

pipe = DiffusionPipeline.from_pretrained(
    "playgroundai/playground-v2.5-1024px-aesthetic",
    torch_dtype=torch.float16,
    variant="fp16",
).to(DEFAULT_DEVICE)


def generate_images(prompts):
    with torch.autocast("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"):  # Mixed precision for speed on GPU
        images = pipe(prompts, num_inference_steps=50, guidance_scale=3).images

        return images
