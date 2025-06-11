from pathlib import Path

import model_loader
import pipeline
import torch
from PIL import Image
from transformers import CLIPTokenizer

DEVICE = "cuda"

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE}")

# tokenizer的檔案在https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/main/tokenizer
tokenizer = CLIPTokenizer("vocab.json", merges_file="merges.txt")
# 可以在這裡下載
# https://huggingface.co/Envvi/Inkpunk-Diffusion/tree/main
# https://huggingface.co/ogkalu/Comic-Diffusion/tree/main
model_file = "inkpunk-diffusion-v1.ckpt"


models = model_loader.preload_models_from_standard_weights(model_file, DEVICE)


# TEXT TO IMAGE
prompt = '''cute Maltese dog, big sparkling eyes, fluffy white fur, sitting on grass, anime style, manga illustration, vibrant colors, clean lineart, highly detailed, soft shading, pastel background, adorable expression, masterpiece, best quality, ultra-detailed, 4k, sharp focus, bright lighting, full body'''
uncond_prompt = ""  # Optional: negative prompt
do_cfg = True
cfg_scale = 8  # min: 1, max: 14
input_image = None  # No image for Text-to-Image
strength = 1  # Use 1 as a default value for Text-to-Image

# IMAGE TO IMAGE
# image_path = "cat.jpg"  # Path to input image
# prompt = "A cat with sunglasses, wearing comfy hat, highly detailed, ultra sharp, cinematic, 100mm lens, 8k resolution."
# uncond_prompt = ""  # Optional: negative prompt
# do_cfg = True
# cfg_scale = 8  # min: 1, max: 14
# input_image = Image.open(image_path)
# strength = 0.8  # Strength to control how much transformation occurs

## SAMPLER
sampler = "ddpm"
num_inference_steps = 50
seed = 42

output_image = pipeline.generate(
    prompt=prompt,
    uncond_prompt=uncond_prompt,
    input_image=input_image,
    strength=strength,
    do_cfg=do_cfg,
    cfg_scale=cfg_scale,
    sampler_name=sampler,
    n_inference_steps=num_inference_steps,
    #seed=seed,
    models=models,
    device=DEVICE,
    idle_device=DEVICE, #原本是直接設定在cuda
    tokenizer=tokenizer,
)

# Combine the input image and the output image into a single image.
# Convert NumPy array to PIL image
output_pil = Image.fromarray(output_image)

# Show the image
output_pil.show()

# Save the image (optional)
output_pil.save("output.png")