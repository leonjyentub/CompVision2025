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
model_file = "Inkpunk-Diffusion-v2.ckpt"


models = model_loader.preload_models_from_standard_weights(model_file, DEVICE)

# %%
'''這個model有
'clip': clip,
'encoder': encoder,
'decoder': decoder,
'diffusion': diffusion
使用torchinfo來看這四個模型的結構

from torchinfo import summary

print("CLIP Model Summary:")
summary(models['clip'], input_size=(1, 77), device=DEVICE)
print("Encoder Summary:")
summary(models['encoder'], input_size=(1, 3, 512, 512), device=DEVICE)
print("Decoder Summary:")
summary(models['decoder'], input_size=(1, 4, 64, 64), device=DEVICE)
print("Diffusion Summary:")
summary(models['diffusion'], input_size=(1, 4, 64, 64), device=DEVICE)
'''

# TEXT TO IMAGE
prompt = '''cute Maltese dog, big sparkling eyes, fluffy white fur, sitting on grass, manga illustration, vibrant colors, clean lineart, highly detailed, soft shading, pastel background, adorable expression, masterpiece, best quality, ultra-detailed'''
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