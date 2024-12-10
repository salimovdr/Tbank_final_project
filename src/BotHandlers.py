from telegram import Update, ReplyKeyboardMarkup, ReplyKeyboardRemove
from telegram.ext import ContextTypes
import os
from src.PhotoStorageManager import PhotoStorageManager


class BotHandlers:
    """Handles commands and messages for the bot."""
    def __init__(self, bot):
        self.bot = bot
        self.UPLOAD_DIR = os.getenv('UPLOAD_DIR')
        self.MAX_PHOTO_HISTORY = os.getenv("MAX_PHOTO_HISTORY")

    def generate_keyboard(self):
        return ReplyKeyboardMarkup([["Последнее"], ["Загрузить", "Сгенерировать"]], resize_keyboard=True)

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat_id = update.effective_chat.id
        self.bot.logger.info(f"User {chat_id} started the bot.")
        await update.message.reply_text(
            "Добро пожаловать! Выберите действие:",
            reply_markup=self.generate_keyboard()
        )

    async def handle_photo(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat_id = update.effective_chat.id
        photo = update.message.photo[-1]  # Highest resolution
        user_dir = os.path.join(self.UPLOAD_DIR, str(chat_id))
        PhotoStorageManager.ensure_directory_exists(user_dir)

        PhotoStorageManager.manage_photos(user_dir, self.MAX_PHOTO_HISTORY)

        file_path = os.path.join(user_dir, "1")
        try:
            file = await photo.get_file()
            await file.download_to_drive(file_path)
            self.bot.logger.info(f"Photo saved at {file_path} for user {chat_id}.")
            await update.message.reply_text("Файл успешно загружен, обрабатываем...")
        except Exception as e:
            self.bot.logger.error(f"Error saving photo for {chat_id}: {e}")
            await update.message.reply_text("Произошла ошибка при сохранении файла. Попробуйте снова.")

    async def handle_last(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat_id = update.effective_chat.id
        user_dir = os.path.join(self.UPLOAD_DIR, str(chat_id))

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

    async def text_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        text = update.message.text
        chat_id = update.message.chat_id

        if text == "Загрузить":
            await update.message.reply_text("Пожалуйста, отправьте файл (изображение) для загрузки.")
        elif text == "Последнее":
            await self.handle_last(update, context)
        else:
            await update.message.reply_text("Неизвестная команда. Пожалуйста, используйте кнопки.")