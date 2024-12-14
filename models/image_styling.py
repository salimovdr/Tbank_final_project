from PIL import Image
from torchvision import transforms
import torch

class ImageStyling():
    
    def __init__(self):
        self.input_type = ['image', 'text']
        self.output_type = ['image']
        self.id = 3
        
        
    def style_transfer(self, content_image: Image, style_image: Image, alpha: float = 1.0) -> Image:
        """
        Выполняет перенос стиля с помощью AdaIN.
        
        Args:
            content_image (PIL.Image): Исходное изображение (контент).
            style_image (PIL.Image): Изображение стиля.
            alpha (float): Интенсивность переноса стиля (от 0 до 1).
            
        Returns:
            PIL.Image: Стилизованное изображение.
        """
        preprocess = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        postprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(content_image.size)
        ])

        content_tensor = preprocess(content_image).unsqueeze(0)
        style_tensor = preprocess(style_image).unsqueeze(0)

        # Пример AdaIN (упрощенный)
        content_mean, content_std = torch.mean(content_tensor, dim=(2, 3)), torch.std(content_tensor, dim=(2, 3))
        style_mean, style_std = torch.mean(style_tensor, dim=(2, 3)), torch.std(style_tensor, dim=(2, 3))

        normalized_content = (content_tensor - content_mean[:, :, None, None]) / content_std[:, :, None, None]
        stylized_tensor = normalized_content * style_std[:, :, None, None] + style_mean[:, :, None, None]
        stylized_tensor = alpha * stylized_tensor + (1 - alpha) * content_tensor

        # Обратное преобразование в PIL.Image
        stylized_image = postprocess(torch.clamp(stylized_tensor.squeeze(0), 0, 1))
        return stylized_image


    def predict(self, query: dict, generate_image_func) -> Image:
        """
        Генерирует изображение стиля с помощью существующей функции и выполняет перенос стиля на входное изображение.
        
        Args:
            input_data (dict): Словарь с ключами:
                - "text": промпт для генерации стиля (str).
                - "image": исходное изображение для стилизации (PIL.Image).
            generate_image_function (callable): Функция для генерации изображения стиля.
        
        Returns:
            PIL.Image: Стилизованное изображение.
        """
        if "text" not in query or not query["text"]:
            raise ValueError("Input data must contain a non-empty 'text' key.")
        if "image" not in query or not isinstance(query["image"], Image):
            raise ValueError("Input data must contain a PIL.Image under the 'image' key.")

        # Получаем промпт и изображение
        text_prompt = query["text"]
        content_image = query["image"]

        # Генерируем стильное изображение с помощью переданной функции
        style_image = generate_image_func({"text": text_prompt, "image": None})

        # Применяем AdaIN
        styled_image = self.style_transfer(content_image, style_image)
        return styled_image