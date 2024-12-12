from PIL import Image, ImageOps
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, \
                      EulerAncestralDiscreteScheduler

class ImageEditing():
    def __init__(self):
        model_id = "timbrooks/instruct-pix2pix"
        self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, 
                                                                           torch_dtype=torch.float16, 
                                                                           safety_checker=None)
        self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipe.scheduler.config)
        
        self.input_type = ['image', 'text']
        self.output_type = ['image']
        self.id = 1

    def predict(self, query: dict) -> Image:
        image = query["image"]
        prompt = query["text"]

        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")

        images = self.pipe(prompt, 
                           image=image, 
                           num_inference_steps=10, 
                           image_guidance_scale=1.2).images

        return images[0]
