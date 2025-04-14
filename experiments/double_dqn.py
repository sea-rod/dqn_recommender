# experiments/double_dqn.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from agents.replay_buffer import ReplayBuffer


class DoubleDQNAgent:
    def __init__(
        self,
        state_size,
        action_size,
        model_cls,
        lr=1e-3,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
        buffer_capacity=10000,
        batch_size=64,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.model = model_cls(state_size, action_size)
        self.target_model = model_cls(state_size, action_size)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.memory = ReplayBuffer(buffer_capacity)

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(
            self.batch_size
        )

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        q_current = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_actions = self.model(next_states).argmax(1)
        q_next = (
            self.target_model(next_states)
            .gather(1, next_actions.unsqueeze(1))
            .squeeze(1)
        )
        q_target = rewards + (1 - dones) * self.gamma * q_next

        loss = self.criterion(q_current, q_target.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())
