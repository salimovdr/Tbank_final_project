from diffusers import StableDiffusionPipeline
import torch
from PIL import Image

class ImageGeneration():

    def __init__(self):
        model_id = "sd-legacy/stable-diffusion-v1-5"
        self.pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)

        self.input_type = ['text']  # PIL
        self.output_type = ['image']
        self.id = 0
        

    def predict(self, input: dict) -> Image:
        prompt = input["text"]
        
        return self.pipe(prompt).images[0]