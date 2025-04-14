import gym
from gym import spaces
import numpy as np


class RecommendationEnv(gym.Env):
    def __init__(self, num_users=100, num_items=50, history_length=10):
        super(RecommendationEnv, self).__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.history_length = history_length

        self.user_profiles = np.random.rand(num_users, history_length)
        self.item_embeddings = np.random.rand(num_items, history_length)

        self.action_space = spaces.Discrete(num_items)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(history_length,), dtype=np.float32
        )

        self.current_user = 0

    def reset(self):
        self.current_user = np.random.randint(0, self.num_users)
        return self.user_profiles[self.current_user]

    def step(self, action):
        user_vector = self.user_profiles[self.current_user]
        item_vector = self.item_embeddings[action]

        similarity = np.dot(user_vector, item_vector) / (
            np.linalg.norm(user_vector) * np.linalg.norm(item_vector) + 1e-8
        )
        reward = float(similarity > 0.8)

        done = True
        next_state = self.reset()

        return next_state, reward, done, {}
