# train/train_dqn.py
import yaml
import torch
import numpy as np
from envs.recommendation_env import RecommendationEnv
from agents.model import QNetwork
from agents.dqn_agent import DQNAgent
from utils.logger import Logger


def main():
    with open("config/training_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    env_cfg = config["env"]
    agent_cfg = config["agent"]
    train_cfg = config["train"]

    env = RecommendationEnv(
        num_users=env_cfg["num_users"],
        num_items=env_cfg["num_items"],
        history_length=env_cfg["history_length"],
    )

    state_size = env_cfg["history_length"]
    action_size = env_cfg["num_items"]

    model = QNetwork(state_size, action_size)
    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        model=model,
        lr=agent_cfg["lr"],
        gamma=agent_cfg["gamma"],
        epsilon=agent_cfg["epsilon_start"],
        epsilon_min=agent_cfg["epsilon_min"],
        epsilon_decay=agent_cfg["epsilon_decay"],
        buffer_capacity=agent_cfg["buffer_capacity"],
        batch_size=agent_cfg["batch_size"],
    )

    logger = Logger()

    for episode in range(train_cfg["episodes"]):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.replay()
            state = next_state
            total_reward += reward

        if episode % agent_cfg["target_update_freq"] == 0:
            agent.update_target_network()

        if episode % train_cfg["log_interval"] == 0:
            print(
                f"Episode {episode}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.3f}"
            )
            logger.log_scalar("reward", total_reward, episode)
            logger.log_scalar("epsilon", agent.epsilon, episode)

    torch.save(agent.model.state_dict(), train_cfg["save_model_path"])
    logger.close()
