from PIL import Image, ImageOps
from diffusers import DiffusionPipeline
import torch

class BackgroundEditing():
    
    def __init__(self):
        model_id = "yahoo-inc/photo-background-generation"
        self.pipe = DiffusionPipeline.from_pretrained(model_id, 
                                                      custom_pipeline=model_id,
                                                      torch_dtype=torch.float16)
        self.seed = 13
        self.cond_scale = 1.0
        self.n_images_per_prompt = 1
        self.n_inference_steps = 30
        
        self.input_type = ['image', 'text']
        self.output_type = ['image']
        self.id = 3

    def predict(self, query: dict, bg_remover) -> Image:
        image = query["image"]
        prompt = query["text"]
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")

        fg_mask = bg_remover.predict({"text": None, "image": image},
                                     map=True)
        mask = ImageOps.invert(fg_mask)
        generator = torch.Generator(device='cpu').manual_seed(self.seed)        

        with torch.no_grad():
            images = self.pipe(prompt=prompt,
                                image=image,
                                mask_image=mask,
                                control_image=mask,
                                num_images_per_prompt=self.n_images_per_prompt,
                                generator=generator,
                                num_inference_steps=self.n_inference_steps,
                                guess_mode=False,
                                controlnet_conditioning_scale=self.cond_scale)
        return images.images[0]