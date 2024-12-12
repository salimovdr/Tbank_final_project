import os
from src.Logger import Logger
from src.PhotoStorageManager import PhotoStorageManager
from telegram.ext import Application, CommandHandler, MessageHandler, filters
from src.BotHandlers import BotHandlers


class PhotoBot:
    """Main bot class."""
    def __init__(self, token):
        self.UPLOAD_DIR = os.getenv('UPLOAD_DIR')
        self.token = token
        self.logger = Logger.setup()
        PhotoStorageManager.ensure_directory_exists(self.UPLOAD_DIR)
        self.application = Application.builder().token(self.token).build()
        self.handlers = BotHandlers(self)
        self._register_handlers()

    def _register_handlers(self):
        self.application.add_handler(CommandHandler("start", self.handlers.start))
        self.application.add_handler(MessageHandler(filters.PHOTO, self.handlers.handle_photo))
        self.application.add_handler(MessageHandler(filters.TEXT, self.handlers.text_handler))

    def run(self):
        self.logger.info("Bot is running.")
        self.application.run_polling()
