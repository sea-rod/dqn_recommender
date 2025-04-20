# envs/recommendation_env.py
import numpy as np
import pandas as pd
import random
from collections import defaultdict


class RecommendationEnv:
    def __init__(
        self,
        history_length=5,
        min_interactions=10,
        data_path="data/processed/data.csv",
    ):
        self.history_length = history_length
        self.min_interactions = min_interactions
        self.data_path = data_path
        self.load_data()

    def load_data(self):
        # Load MovieLens data
        df = pd.read_csv(self.data_path)

        # Filter users with enough interactions
        user_counts = df["userId"].value_counts()
        valid_users = user_counts[user_counts >= self.min_interactions].index
        df = df[df["userId"].isin(valid_users)]

        self.user_item_dict = defaultdict(list)
        for _, item_id, user_id, rating, _ in df.itertuples(index=False):
            self.user_item_dict[user_id].append((item_id, rating))

        self.all_items = list(df["movieId"].unique())
        self.num_items = max(self.all_items) + 1
        self.users = list(self.user_item_dict.keys())
        self.current_user = None
        self.interaction_index = None

    def reset(self):
        # Pick a random user with enough history
        while True:
            self.current_user = random.choice(self.users)
            user_interactions = self.user_item_dict[self.current_user]
            if len(user_interactions) > self.history_length:
                break

        self.interaction_index = self.history_length
        self.state = [item for item, _ in user_interactions[: self.history_length]]
        return np.array(self.state)

    def step(self, action):
        user_interactions = self.user_item_dict[self.current_user]

        if self.interaction_index >= len(user_interactions):
            done = True
            return np.array(self.state), 0.0, done, {}

        actual_item, rating = user_interactions[self.interaction_index]
        reward = 0.5 if actual_item in action else 0.0
        reward += 1.0 if rating >= 4 else 0.0

        self.state = self.state[1:] + [action[0]]
        self.interaction_index += 1
        done = self.interaction_index >= len(user_interactions)

        return np.array(self.state), reward, done, {}

    def get_action_space(self):
        return self.num_items
