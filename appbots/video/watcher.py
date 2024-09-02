import random
import time

from appbots.core.mqtt import pub_topic, mqtt_runner

bot_device = "192.168.10.112"


def pub_command(action: str, *args):
    pub_topic("command", {"action": action, "device": bot_device, "args": args})


def launch(name: str):
    pub_command("launch", name)


def swipe_up():
    pub_command("swipe_up")


def run():
    counter = 0
    duration = 6

    launch("百度极速版")

    while True:
        counter += 1
        swipe_up()

        print(f"swipe {counter} done")

        time.sleep(duration)

        if counter > int(60 / duration) * 65:
            break


if __name__ == "__main__":
    mqtt_runner(lambda : {
        # launch("抖音")
        run()
    })
