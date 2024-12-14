from telegram import Update, ReplyKeyboardMarkup, ReplyKeyboardRemove
from telegram.ext import ContextTypes
from src.PhotoStorageManager import PhotoStorageManager
from src.JsonLogger import JsonLogger
import shutil

import os
from pathlib import Path
import json
import io

import ollama
from ollama import Options

import torch
from PIL import Image

# models
from models.image_editing import ImageEditing
from models.bg_removing import BackgroundRemover
from models.bg_editing import BackgroundEditing
from models.image_generation import ImageGeneration

from deep_translator import GoogleTranslator


# useful functions------------------------------------------------------------------------

def pil_to_bytes(pil_image: Image) -> io.BytesIO:
    """Converts a PIL Image to a BytesIO object.

    Args:
        pil_image: The PIL Image object.

    Returns:
        A BytesIO object containing the image data.
    """
    img_io = io.BytesIO()
    pil_image.save(img_io, 'PNG')  # Or 'PNG', or another suitable format
    img_io.seek(0)  # Reset the stream pointer to the beginning
    return img_io


def read_file(input_path: str) -> str:
    "reads file specified by input_path"
    file_path = Path(input_path)
    with file_path.open('r', encoding='utf-8') as f:
        text = f.read()
    return text


def prepare_history(hst: list) -> str:
    "prepares telegram history for LLM input"
    hst4model = [a['text'] for a in hst if a['user'] != 'model']
    return str(hst4model)

# ----------------------------------------------------------------------------------------
# bot class ------------------------------------------------------------------------------


class BotHandlers:
    """Handles commands and messages for the bot."""
    def __init__(self, bot):
        self.bot = bot
        self.UPLOAD_DIR = os.getenv('UPLOAD_DIR')
        self.MAX_PHOTO_HISTORY = os.getenv("MAX_PHOTO_HISTORY")
        self.JsonLogger = JsonLogger(os.getenv("MAX_MESSAGE_HISTORY"), os.getenv("MAX_PHOTO_HISTORY"))

        # environmental variables
        self.OLLAMA_CONNECTION_STR = os.environ.get(
            "OLLAMA_CONNECTION_STR", "http://localhost:11434"
        )  # local url
        self.OLLAMA_MODEL = os.environ.get("MODEL_NAME")  # ollama model name

        self.TEMPERATURE = float(os.environ.get("TEMPERATURE", 0.3))  # temperature for json generation
        self.CONTEXT_LENGTH = int(os.environ.get("CONTEXT_LENGTH", 4096))  # input context length
        self.MAX_TOKENS = int(os.environ.get("MAX_TOKENS", 512))  # max tokens for output
        self.BASE_WIDTH = int(os.environ.get("BASE_WIDTH", 300))

        self.MODEL_LIST_PATH = os.environ.get("MODEL_LIST_PATH", "prompts/model_list.json")
        self.JSON_PROMPT_PATH = os.environ.get("JSON_PROMPT_PATH", "prompts/json_prompt.txt")
        self.JSON_SCHEMA_PATH = os.environ.get("JSON_SCHEMA_PATH", "prompts/json_schema.txt")
        self.IMG_FOLDER = os.environ.get("IMG_FOLDER", "storage/")

        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.img_edit_model = ImageEditing()
        self.img_gen_model = ImageGeneration()
        self.bg_remover = BackgroundRemover()
        self.bg_editor = BackgroundEditing()

        self.img_edit_model.pipe.to(self.DEVICE)
        self.img_gen_model.pipe.to(self.DEVICE)
        self.bg_editor.pipe.to(self.DEVICE)
        
        self.MODELS = {0: self.img_gen_model,
                       1: self.img_edit_model,
                       2: self.bg_remover,
                       3: self.bg_editor,}

        self.translator = GoogleTranslator(source='auto', target='en')

        torch.cuda.empty_cache()
        
        
    # bot functions-----------------------------------------------------------------------
    # ------------------------------------------------------------------------------------
    
    def generate_keyboard(self):
        return ReplyKeyboardMarkup(["Очистить"], resize_keyboard=True)   
    
    
    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat_id = update.effective_chat.id
        self.bot.logger.info(f"User {chat_id} started the bot.")
        await update.message.reply_text(
            "Добро пожаловать! Выберите действие:",
            reply_markup=self.generate_keyboard()
        )  
    
    """
    async def handle_photo_and_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        self.bot.logger.info(f"HANDLE_PHOTO_AND_TEXT")
        
        await self.handle_photo(update, context)
        await self.handle_text(update, context)
    """
    
    
    async def handle_photo(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat_id = update.effective_chat.id
        photo = update.message.photo[-1] # Самое высокое разрешение
        caption = update.message.caption
        user_dir = os.path.join(self.UPLOAD_DIR, str(chat_id), "user")
        
        PhotoStorageManager.ensure_directory_exists(user_dir)
        PhotoStorageManager.manage_photos(user_dir, self.MAX_PHOTO_HISTORY)

        img_id = str(len(os.listdir(user_dir)) + 1)
        file_path = os.path.join(user_dir, img_id)
        try:
            file = await photo.get_file()
            await file.download_to_drive(file_path)
            self.bot.logger.info(f"Photo saved at {file_path} for user {chat_id}.")
        except Exception as e:
            await update.message.reply_text("Произошла ошибка при сохранении фото.")
            
        
        if caption:
            await self.handle_text(update, context, caption=True)
            
    
    async def handle_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE, caption=False) -> None:
        if caption:
            text = update.message.caption
        else:
            text = update.message.text
        chat_id = update.message.chat_id
        
        if text == "Очистить":
            try:
                path_dir = os.path.join(self.UPLOAD_DIR, str(chat_id))
                if os.path.exists(path_dir) and os.path.isdir(path_dir):
                    shutil.rmtree(path_dir)  # Удаляет папку и всё её содержимое
            except Exception as e:
                await update.message.reply_text("Произошла ошибка при удалении истории.")
        else:
            "update history"
            try:
                messages_path = os.path.join(self.UPLOAD_DIR, str(chat_id), "messages.json")
                self.JsonLogger.add(messages_path, str(chat_id), text, -1)
            except Exception as e:
                await update.message.reply_text("Произошла ошибка при обновлении истории.")
            
            "run the script"
            history = self.JsonLogger.get(messages_path)
            try:
                history = list(history)
            except Exception as e:
                print(f"Произошла ошибка: {e}")
            
            res = await self.model_say(history)
            
            if res and 'text' in res and res['text']:
                await update.message.reply_text(res['text'])
            elif not res or 'text' not in res:
                await update.message.reply_text("No output or 'text' key not found in model response.")
                
            if res and 'image' in res and res['image']:
                try:
                    user_dir = os.path.join(self.UPLOAD_DIR, str(chat_id), "user")
                    PhotoStorageManager.ensure_directory_exists(user_dir)
                    PhotoStorageManager.manage_photos(user_dir, self.MAX_PHOTO_HISTORY)
                    img_id = str(len(os.listdir(user_dir)) + 1)
                    file_path = os.path.join(user_dir, img_id)
                    res['image'].save(file_path, 'PNG')
                    self.bot.logger.info(f"Generated image saved at {file_path} for user {chat_id}.")
                except Exception as e:
                    await update.message.reply_text("Error occured while saving generated image.")
                    
                bytes_img = pil_to_bytes(res['image'])
                await update.message.reply_photo(bytes_img)
            elif not res or 'image' not in res:
                await update.message.reply_text("No output or 'image' key not found in model response.")
                
    
    # ------------------------------------------------------------------------------------
    # main function ----------------------------------------------------------------------

    async def model_say(self, history: list) -> dict:
        """
        работает модель но если прям в функции будет выполняться все ляжет((

        hotoStorageManager.ensure_directory_exists(user_dir)
        text img = To_MODEL(self.JsonLogger.get(messages_dir))
        if img != -1:
            PhotoStorageManager.manage_photos(model_dir, self.MAX_PHOTO_HISTORY)
            file_path = os.path.join(model_dir, "1")
            with open(file_path, 'wb') as file:
                file.write(data)
            img = 1
        self.JsonLogger.add(messages_dir, "model", text)
        """
        llm_output = self.generate_json(history)
        
        user_id = history[-1]["user"]
        user_dir = os.path.join(self.UPLOAD_DIR, str(user_id), "user")
        
        PhotoStorageManager.ensure_directory_exists(user_dir)
        PhotoStorageManager.manage_photos(user_dir, self.MAX_PHOTO_HISTORY)
        
        if image_id == -1 or len(user_dir) == 0:
            image_id == None
            image = None
        else:
            image_id = os.listdir(user_dir)[-1]
            image = Image.open(os.path.join(user_dir, str(image_id)))
            
        json_output = json.loads(llm_output)
        img_out = None

        chosen_model = self.get_chosen_model(json_output)
        model_id = chosen_model["id"]
        
        bgr = None
        if model_id == 3:
            bgr = self.bg_remover
            
        if model_id in [1, 2, 3] and image is None and len(user_dir) == 0:
            return {"text": "Вначале загрузите изображение", "image": None}
    
        img_out = self.model_handler(self.MODELS[model_id], json_output, image, translate=True, bg_remover=bgr)
        return {"text": llm_output, "image": img_out}

    # ------------------------------------------------------------------------------------

    # submain functions ------------------------------------------------------------------

    def model_handler(self, model, json_output, img_dir, image=None, translate=True, bg_remover=None):
        img_type = json_output["image_type"]
        text = json_output["prompt"]
        if translate:
            query = self.translator.translate(text) 
            
        input_image = None
        
        if len(os.listdir(img_dir)) == 0:
            input_image = image
        else:
            if img_type == "previous" and len(os.listdir(img_dir)) >= 2:
                filename = os.listdir(img_dir)[-2]
            else:
                filename = os.listdir(img_dir)[-1]
            
            input_image = Image.open(os.path.join(img_dir, filename))
            
        input_image = self.resize(input_image)    
        if bg_remover:
            img_out = model.predict({"text": query, "image": input_image}, bg_remover)
        else:
            img_out = model.predict({"text": query, "image": input_image}, bg_remover)
        torch.cuda.empty_cache()
        return img_out


    def generate_json(self, history: list) -> dict:
        """
        Generates json of which model to use

        Args:
        request (Request): {"user_message": text to summarize, 
                            "img": img_id,
                            "history": history}

        Returns:
        dict: {"model_id": model id,
                "prompt": prompt,
                "image_type": 'no', 'last', or 'previous'}
        """
        # read arguments
        user_message = history[-1]["text"]
        hst = prepare_history(history[:-1])
        
        prompt = read_file(self.JSON_PROMPT_PATH)
        json_schema = read_file(self.JSON_SCHEMA_PATH)
        model_list = read_file(self.MODEL_LIST_PATH)
        
        prompt = prompt.replace("$USER_MSG", user_message)  # add user message
        prompt = prompt.replace("$SCHEMA", json_schema)  # add schema
        prompt = prompt.replace("$MODEL_LIST", model_list)  # add model list
        prompt = prompt.replace("$HISTORY", hst)  # add history
        
        api_response = ollama.generate(
                        model=self.OLLAMA_MODEL,
                        prompt=prompt,
                        format='json',
                        stream=False,
                        options=Options(
                            num_ctx=self.CONTEXT_LENGTH,
                            num_predict=-1,
                            temperature=self.TEMPERATURE,
            ),
        )
        response = api_response["response"]
        return response

    # ------------------------------------------------------------------------------------

    # additional functions ---------------------------------------------------------------

    def resize(self, img: Image) -> Image:
        "resize image to width = BASE_WIDTH"
        ratio = (self.BASE_WIDTH / float(img.size[0]))
        h_size = int((float(img.size[1]) * float(ratio)))
        
        return img.resize((self.BASE_WIDTH, h_size), Image.Resampling.LANCZOS)


    def get_chosen_model(self, llm_response: dict) -> dict:
        """
        finds the model from model_list based on its id
        
        Args:
            llm_response (dict): {"model_id": model id,
                                  "prompt": prompt,
                                  "image_type": 'no', 'last', or 'previous'}
        Returns:
            dict: {"id": model id,
                "name": model name,
                "description": what this model do}
        """
        model_id = llm_response["model_id"]
        with open(self.MODEL_LIST_PATH, 'r', encoding='utf-8') as f:
            models = json.load(f)['models']   
        chosen_model = None
        
        for m in models:
            if m["id"] == model_id:
                chosen_model = m
        
        if chosen_model is None:
            raise Exception(404, "LLM refers to unknown model")
        return chosen_model

    # ------------------------------------------------------------------------------------
