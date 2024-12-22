from transparent_background import Remover
from PIL import Image

class BackgroundRemover():

    def __init__(self):
        self.pipe = Remover(mode='base')
        self.input_type = ['image']  # PIL
        self.output_type = ['image']
        self.id = 2
        

    def predict(self, query: dict, map=False) -> Image:
        image = query["image"]
        
        if map:
            return self.pipe.process(image, type='map')
        
        return self.pipe.process(image)