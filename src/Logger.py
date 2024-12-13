import logging


class Logger:
    """Logger setup for the application."""
    @staticmethod
    def setup():
        logging.basicConfig(
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            level=logging.INFO
        )
        return logging.getLogger(__name__)