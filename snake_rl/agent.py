from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Deque, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from snake_rl.model import LinearQNet


@dataclass
class AgentConfig:
    gamma: float = 0.97
    lr: float = 0.0007
    max_memory: int = 100_000
    batch_size: int = 2048
    epsilon_start: float = 1.0
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.9975
    target_sync_interval: int = 250
    train_start_size: int = 2000
    grad_clip_norm: float = 5.0


class DQNAgent:
    def __init__(self, state_size: int = 11, action_size: int = 3, hidden_size: int = 256, config: AgentConfig | None = None) -> None:
        self.config = config or AgentConfig()
        self.state_size = state_size
        self.action_size = action_size

        self.epsilon = self.config.epsilon_start
        self.memory: Deque[Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]] = deque(maxlen=self.config.max_memory)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LinearQNet(state_size, hidden_size, action_size).to(self.device)
        self.target_model = LinearQNet(state_size, hidden_size, action_size).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr)
        self.loss_fn = nn.SmoothL1Loss()
        self.train_steps = 0

    def remember(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: bool) -> None:
        self.memory.append((state, action, reward, next_state, done))

    def get_action(self, state: np.ndarray, train: bool = True) -> np.ndarray:
        if train and random.random() < self.epsilon:
            move = random.randint(0, self.action_size - 1)
            action = np.zeros(self.action_size, dtype=int)
            action[move] = 1
            return action

        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            prediction = self.model(state_tensor)
        move = int(torch.argmax(prediction).item())
        action = np.zeros(self.action_size, dtype=int)
        action[move] = 1
        return action

    def train_short_memory(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: bool) -> float:
        return self._train_step([state], [action], [reward], [next_state], [done])

    def train_long_memory(self) -> float:
        if len(self.memory) < self.config.train_start_size:
            return 0.0

        sample_size = min(len(self.memory), self.config.batch_size)
        mini_sample = random.sample(self.memory, sample_size)

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        return self._train_step(states, actions, rewards, next_states, dones)

    def train_from_replay(self) -> float:
        return self.train_long_memory()

    def _train_step(self, states, actions, rewards, next_states, dones) -> float:
        states_tensor = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        next_states_tensor = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        actions_tensor = torch.tensor(np.array(actions), dtype=torch.float32, device=self.device)
        rewards_tensor = torch.tensor(np.array(rewards), dtype=torch.float32, device=self.device)
        dones_tensor = torch.tensor(np.array(dones), dtype=torch.bool, device=self.device)

        pred = self.model(states_tensor)
        target = pred.clone().detach()

        with torch.no_grad():
            # Double DQN: use online network to select action, target network to evaluate action value.
            online_next_q = self.model(next_states_tensor)
            best_next_actions = torch.argmax(online_next_q, dim=1)
            target_next_q = self.target_model(next_states_tensor)
            max_next_q = target_next_q[torch.arange(target_next_q.size(0)), best_next_actions]

            q_new = rewards_tensor + (~dones_tensor).float() * self.config.gamma * max_next_q

        action_indices = torch.argmax(actions_tensor, dim=1)
        target[torch.arange(target.size(0)), action_indices] = q_new

        self.optimizer.zero_grad()
        loss = self.loss_fn(pred, target)
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
        self.optimizer.step()

        self.train_steps += 1
        if self.train_steps % self.config.target_sync_interval == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        return float(loss.item())

    def update_epsilon(self) -> None:
        self.epsilon = max(self.config.epsilon_min, self.epsilon * self.config.epsilon_decay)

    def save(self, file_path: str) -> None:
        self.model.save(file_path)

    def load(self, file_path: str) -> None:
        self.model.load(file_path, map_location=str(self.device))
        self.target_model.load_state_dict(self.model.state_dict())
