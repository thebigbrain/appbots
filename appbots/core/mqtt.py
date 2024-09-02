import json

import paho.mqtt.client as mqtt


# The callback for when the client receives a CONNACT response from the server.
def on_connect(client, userdata, flags, reason_code, properties):
    print(f"Connected with result code {reason_code}")
    # Subscribing in on_connect() means that if we lose the connection and
    #  reconnect, then subscriptions will be renewed.
    client.subscribe("$SYS/#")


# The callback for when a PUBLISH message is received from the server.
def on_message(client, userdata, msg):
    if str(msg.topic).startswith("$SYS/"):
        return

    print(f"Message Arrived: topic<{msg.topic}>  {msg.payload.decode('utf-8')}")


mqttc = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
mqttc.on_connect = on_connect
mqttc.on_message = on_message

mqttc.connect(host="192.168.10.115")


def pub_topic(topic, payload):
    msg = mqttc.publish(topic, payload=json.dumps(payload), qos=1)
    msg.wait_for_publish(timeout=3)


def start_mqtt_loop():
    mqttc.loop_start()


def stop_mqtt_loop():
    mqttc.disconnect()
    mqttc.loop_stop()


def mqtt_runner(task):
    start_mqtt_loop()
    task()
    stop_mqtt_loop()


def start_bot(name):
    pub_topic("automator/bot", {"method": "start", "params": {"name": name}})


def stop_bot():
    pub_topic("automator/bot", {"method": "stop", })


if __name__ == "__main__":
    # stop_bot()

    # start_bot("快手")
    mqtt_runner(lambda : {
        start_bot("速读免费小说")
    })

