import time

from appbots.core import actions
from appbots.core.mqtt import mqtt_runner


def run():
    counter = 0
    duration = 6

    actions.launch("百度极速版")

    while True:
        time.sleep(duration)

        counter += 1
        actions.swipe_up()

        print(f"swipe {counter} done")

        if counter > int(60 / duration) * 65:
            break


if __name__ == "__main__":
    mqtt_runner(lambda: {
        # actions.launch('百度极速版')
        run()
    })
