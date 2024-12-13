from telegram import Update, ReplyKeyboardMarkup, ReplyKeyboardRemove
from telegram.ext import ContextTypes
from src.PhotoStorageManager import PhotoStorageManager
from src.JsonLogger import JsonLogger
import shutil

import os
from pathlib import Path
import json

import ollama
from ollama import Options

import torch
from PIL import Image

from models.image_editing import ImageEditing
from models.bg_removing import BackgroundRemover

from deep_translator import GoogleTranslator



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

        self.HISTORY = []  # TO CHANGE !!!
        self.IMG_FOLDER = 'images/'  # TO CHANGE !!!
        self.IMG_CNT = len(os.listdir(self.IMG_FOLDER[:-1]))  # TO CHANGE !!!

        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.img_edit_model = ImageEditing()
        self.img_edit_model.pipe.to(self.DEVICE)
        self.bg_remover = BackgroundRemover()

        self.translator = GoogleTranslator(source='auto', target='en')

        torch.cuda.empty_cache()

        """
        img_gen_model = ImageGeneration()
        img_gen_model.pipe.to("cpu")

        torch.cuda.empty_cache()
        """


    def generate_keyboard(self):
        return ReplyKeyboardMarkup([["Последнее"], ["Очистить", "Сгенерировать"]], resize_keyboard=True)

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat_id = update.effective_chat.id
        self.bot.logger.info(f"User {chat_id} started the bot.")
        await update.message.reply_text(
            "Добро пожаловать! Выберите действие:",
            reply_markup=self.generate_keyboard()
        )

    async def handle_photo(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat_id = update.effective_chat.id
        photo = update.message.photo[-1]  # Самое высокое разрешение
        caption = update.message.caption  # Текст, отправленный вместе с фото
        user_dir = os.path.join(self.UPLOAD_DIR, str(chat_id), "user")
        PhotoStorageManager.ensure_directory_exists(user_dir)

        PhotoStorageManager.manage_photos(user_dir, self.MAX_PHOTO_HISTORY)

        file_path = os.path.join(user_dir, "1")
        try:
            file = await photo.get_file()
            await file.download_to_drive(file_path)

            self.bot.logger.info(f"Photo saved at {file_path} for user {chat_id}.")

            messages_dir = os.path.join(self.UPLOAD_DIR, str(chat_id), "messages.json")
            # Логируем текстовое сообщение вместе с фото
            if caption:
                #self.JsonLogger.add(user_dir, "user", caption)
                self.bot.logger.info(f"Caption logged: {caption}")
                self.JsonLogger.add(messages_dir, str(chat_id), caption, 1)
            else:

                self.JsonLogger.add(messages_dir, str(chat_id), "", 1)

            """модель работает"""
            await update.message.reply_text(self.model_say(self.JsonLogger.get(messages_dir)).get("text"))
            #await update.message.reply_photo(photo=file, caption=f"Фото")
        except Exception as e:
            #self.bot.logger.error(f"Error saving photo for {chat_id}: {e}")
            await update.message.reply_text("Произошла ошибка при сохранении файла. Попробуйте снова.")

    async def text_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        text = update.message.text
        chat_id = update.message.chat_id
        if text == "Очистить":
            path_dir = os.path.join(self.UPLOAD_DIR, str(chat_id))
            if os.path.exists(path_dir) and os.path.isdir(path_dir):
                shutil.rmtree(path_dir)  # Удаляет папку и всё её содержимое
        elif text == "Последнее":
            await self.handle_last_img(update, context)
        else:
            messages_dir = os.path.join(self.UPLOAD_DIR, str(chat_id), "messages.json")
            self.JsonLogger.add(messages_dir, str(chat_id), text, -1)
            """"модель работает"""
            await update.message.reply_text(self.model_say(self.JsonLogger.get(messages_dir)).get("text"))
            print(self.JsonLogger.get(messages_dir))

            #await update.message.reply_text("Неизвестная команда. Пожалуйста, используйте кнопки.")

# ------------------

    def resize(self, img: Image) -> Image:
        ratio = (self.BASE_WIDTH / float(img.size[0]))
        h_size = int((float(img.size[1]) * float(ratio)))

        return img.resize((self.BASE_WIDTH, h_size), Image.Resampling.LANCZOS)


    def model_handler(self, model, json_output, image=None, translate=True):
        img_type = json_output["image_type"]
        text = json_output["prompt"]
        if translate:
            query = self.translator.translate(text)

        input_image = None

        if len(os.listdir(self.IMG_FOLDER[:-1])) == 0:
            input_image = image
        else:
            if img_type == "previous" and len(os.listdir(self.IMG_FOLDER[:-1])) >= 2:
                filename = os.listdir(self.IMG_FOLDER[:-1])[-2]
            else:
                filename = os.listdir(self.IMG_FOLDER[:-1])[-1]
            input_image = Image.open(self.IMG_FOLDER + filename)

        input_image = self.resize(input_image)

        img_out = model.predict({"text": query, "image": input_image})
        torch.cuda.empty_cache()
        return img_out


    def read_file(self, input_path: str) -> str:
        file_path = Path(input_path)
        with file_path.open('r', encoding='utf-8') as f:
            text = f.read()
        return text


    def generate_json(self, payload: dict) -> dict:
        """
        Generates .json version of summary

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
        user_message = payload["user_message"]
        user_img_id = payload["user_img"]

        # history = payload.get("history")  CHANGE LATER !!!

        prompt = self.read_file(self.JSON_PROMPT_PATH)
        json_schema = self.read_file(self.JSON_SCHEMA_PATH)
        model_list = self.read_file(self.MODEL_LIST_PATH)

        hst = self.prepare_history(self.HISTORY)

        prompt = prompt.replace("$USER_MSG", user_message)  # add user message
        prompt = prompt.replace("$SCHEMA", json_schema)  # add schema
        prompt = prompt.replace("$MODEL_LIST", model_list)  # add model list
        prompt = prompt.replace("$HISTORY", hst)  # add history  CHANGE LATER !!!

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

        self.update_history_user(user_message, user_img_id)
        self.update_history_model(json.loads(response))

        # logs
        print(hst, end='\n\n')
        return response


    def save_image(self, img: Image):
        num_files = len(os.listdir(self.IMG_FOLDER))
        img.save(self.IMG_FOLDER + str(num_files) + '.png')


    def prepare_history(self, hst):
        hst4model = [a['user_message'] for a in hst if 'user_message' in a.keys()]
        return str(hst4model)


    def get_chosen_model(self, llm_response):
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


    def update_history_user(self, user_msg="-", image_id="-"):
        if user_msg is None:
            user_msg = "-"
        if image_id is None:
            image_id = "-"

        if image_id != "-":
            self.IMG_CNT += 1

        self.HISTORY.append({
            "user_message": user_msg,
            "image_id": image_id,
        }, )


    def update_history_model(self, llm_response):
        chosen_model = self.get_chosen_model(llm_response)

        history_sample = {'model_message': "-",
                          'image_id': "-"}

        if chosen_model["output"] == "image":
            history_sample["image_id"] = self.IMG_CNT
            self.IMG_CNT += 1

        self.HISTORY.append(history_sample)


    def model_say(self, payload):
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
        self.JsonLogger.add(messages_dir, "model", text, img)
        """

        text = payload.get("text")
        image = payload.get("image")
        history = payload.get("history")

        if text == "\\reset":
            self.HISTORY = []
            return "History deleted", None

        if text == "\\rm images":
            return "TODO add function that clears img folder", None

        image_id = self.IMG_CNT if image else None
        llm_output = self.generate_json({"user_message": text,
                                    "user_img": image_id})

        if image:
            self.save_image(image)

        json_output = json.loads(llm_output)
        img_out = None

        chosen_model = self.get_chosen_model(json_output)
        model_id = chosen_model["id"]

        if model_id in [1, 2]:
            models = {1: self.img_edit_model,
                      2: self.bg_remover}
            img_out = self.model_handler(models[model_id], json_output, image, translate=True)
            self.save_image(img_out)

        return {"text": llm_output, "image": img_out}


    async def handle_last_img(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat_id = update.effective_chat.id
        user_dir = os.path.join(self.UPLOAD_DIR, str(chat_id), "user")

        if os.path.exists(user_dir) and os.listdir(user_dir):
            files = sorted([f for f in os.listdir(user_dir) if f.isdigit()], key=int)
            for file_name in files:
                file_path = os.path.join(user_dir, file_name)
                with open(file_path, "rb") as file:
                    await update.message.reply_photo(photo=file, caption=f"Фото {file_name}")
            context.user_data["awaiting_photo_selection"] = True
            await update.message.reply_text(
                f"Выберите изображение, с которым хотите работать, введя его номер (1 - {self.MAX_PHOTO_HISTORY}).",
                reply_markup=ReplyKeyboardRemove()
            )
        else:
            await update.message.reply_text("Файлы не найдены. Загрузите файлы с помощью кнопки 'Загрузить'.")