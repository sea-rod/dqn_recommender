# train/evaluate.py
import torch
import numpy as np
from envs.recommendation_env import RecommendationEnv
from agents.model import QNetwork
from utils.metrics import precision_at_k, ndcg_at_k


def evaluate(model_path, episodes=100, top_k=5):
    env = RecommendationEnv()
    state_size = env.history_length
    action_size = env.num_items

    model = QNetwork(state_size, action_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    precision_list = []
    ndcg_list = []

    for _ in range(episodes):
        state = env.reset()
        done = False
        while not done:
            with torch.no_grad():
                q_values = model(torch.FloatTensor(state).unsqueeze(0))
                top_actions = (
                    torch.topk(q_values, top_k, dim=1).indices.squeeze(0).cpu().numpy()
                )

            next_state, reward, done, _ = env.step(top_actions)

            # True item the user interacted with
            actual_item = env.user_item_dict[env.current_user][
                env.interaction_index - 1
            ][0]
            relevant_items = [actual_item]  # ground truth item

            precision = precision_at_k(top_actions, relevant_items, top_k)
            ndcg = ndcg_at_k(top_actions, relevant_items, top_k)

            precision_list.append(precision)
            ndcg_list.append(ndcg)

            state = next_state

    print(f"Avg Precision@{top_k}: {np.mean(precision_list):.4f}")
    print(f"Avg nDCG@{top_k}: {np.mean(ndcg_list):.4f}")


if __name__ == "__main__":
    evaluate("models/dqn_model.pth")
