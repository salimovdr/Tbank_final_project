import asyncio

from telegram import Update, ReplyKeyboardMarkup, ReplyKeyboardRemove
from telegram.constants import ChatAction
from telegram.ext import ContextTypes
import os
from src.PhotoStorageManager import PhotoStorageManager
from src.JsonLogger import JsonLogger
import shutil
import time


class BotHandlers:
    """Handles commands and messages for the bot."""

    def __init__(self, bot):
        self.bot = bot
        self.UPLOAD_DIR = os.getenv('UPLOAD_DIR')
        self.MAX_PHOTO_HISTORY = os.getenv("MAX_PHOTO_HISTORY")
        self.JsonLogger = JsonLogger(os.getenv("MAX_MESSAGE_HISTORY"), os.getenv("MAX_PHOTO_HISTORY"))

    def generate_keyboard(self):
        return ReplyKeyboardMarkup([["Очистить"]], resize_keyboard=True)

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
                # self.JsonLogger.add(user_dir, "user", caption)
                self.bot.logger.info(f"Caption logged: {caption}")
                self.JsonLogger.add(messages_dir, str(chat_id), caption, 1)
            else:

                self.JsonLogger.add(messages_dir, str(chat_id), "", 1)
                typing_task = asyncio.create_task(self.send_typing_status(chat_id, context))
                await asyncio.sleep(5)
                text = self.model_say()
                # Завершаем отправку статуса
                typing_task.cancel()
                await typing_task  # Ждем завершения задачи, если она не была отменена
                """модель работает"""
                await update.message.reply_text(text)
                # await update.message.reply_photo(photo=file, caption=f"Фото")
        except Exception as e:
            # self.bot.logger.error(f"Error saving photo for {chat_id}: {e}")
            await update.message.reply_text("Произошла ошибка при сохранении файла. Попробуйте снова.")

    # Запускаем асинхронную функцию для периодической отправки статуса
    async def send_typing_status(self, chat_id: int, context: ContextTypes.DEFAULT_TYPE):
        print("шпгнгиитгр")
        try:
            while True:
                await context.bot.send_chat_action(chat_id, ChatAction.TYPING)
                await asyncio.sleep(5)  # Отправляем статус каждые 5 секунд
        except asyncio.CancelledError:
            pass  # Задача была отменена, просто завершаем выполнение

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

            typing_task = asyncio.create_task(self.send_typing_status(chat_id, context))
            await asyncio.sleep(5)
            # Здесь ваша логика обработки фото, например, вызов модели
            model_response = self.model_say()  # Предполагается, что у вас есть метод для обработки

            # Завершаем отправку статуса
            typing_task.cancel()
            await typing_task  # Ждем завершения задачи, если она не была отменена

            # Отправляем ответ от модели
            await update.message.reply_text(model_response)
        """"модель работает"""
        print(self.JsonLogger.get(messages_dir))

        # await update.message.reply_text("Неизвестная команда. Пожалуйста, используйте кнопки.")

    # ------------------

    def model_say(self):
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
        text = "типо модель что то сказала"
        return text

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
