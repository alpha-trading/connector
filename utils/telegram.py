import os

from telegram import Bot


class Telegram:
    def __init__(self, token: str = None, chat_id: str = None):
        if not token:
            token = os.environ.get('TELEGRAM_TOKEN')
        if not chat_id:
            chat_id = os.environ.get('TELEGRAM_CHAT_ID')
        self.chat_id = chat_id
        self.bot = Bot(token)

    def send(self, message: str, **kwargs: dict):
        self.bot.send_message(self.chat_id, message, **kwargs)
