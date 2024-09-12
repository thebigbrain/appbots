import logging

logging.basicConfig(
    format="{asctime} - {name}[{levelname}] - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M",
)


def get_logger(name: str, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.level = level
    return logger
