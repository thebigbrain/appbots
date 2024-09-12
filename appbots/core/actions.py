from appbots.core.mqtt import pub_topic

bot_device = "192.168.10.112"


def pub_command(action: str, *args):
    pub_topic("command", {"action": action, "device": bot_device, "args": args})


def launch(name: str):
    pub_command("launch", name)


def un_launch():
    pub_command("launch", "")


def swipe_up():
    pub_command("swipe_up")


def take_screenshot():
    pub_command("take_screenshot")
