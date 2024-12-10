import os
from dotenv import load_dotenv
from src.PhotoBot import PhotoBot


if __name__ == "__main__":
    load_dotenv()
    bot_token = os.getenv("TG_TOKEN")
    bot = PhotoBot(bot_token)
    bot.run()