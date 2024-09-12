import time

from appbots.core import actions
from appbots.core.logger import get_logger
from appbots.core.mqtt import mqtt_runner

logger = get_logger("screenshot")


def run():
    actions.un_launch()
    while True:
        time.sleep(10)
        actions.take_screenshot()
        logger.info("take screenshot")


if __name__ == "__main__":
    mqtt_runner(run)
