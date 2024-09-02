import time

from appbots.core.time_util import TimeConsuming
from appbots.uibot.agent import UiAgent
from appbots.core.agent import Agent
from appbots.core.env import Env
from appbots.uibot.env import UiElementEnv

env: Env = UiElementEnv()
agent: Agent = UiAgent("app-bot", env)

target_update = 100

# 训练循环
for episode in range(1000):
    state, info = env.reset()
    print(f"start episode {episode} ...")
    for t in range(1000):
        time_consuming = TimeConsuming()

        action = agent.get_action(state)
        next_state, reward, terminated, trunked, info = env.step(action)
        done = terminated or trunked

        agent.update(state, action, reward, done, next_state)

        state = next_state

        if done:
            break

        if episode > 0 and episode % target_update == 0:
            agent.save()

        _, elapsed_text = time_consuming.elapsed()
        print(f"episode {episode}, step {t}, reward {reward}, action {action}, {elapsed_text}")


if __name__ == '__main__':
    pass
