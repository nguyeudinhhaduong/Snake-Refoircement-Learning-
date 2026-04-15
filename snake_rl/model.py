from __future__ import annotations

import torch
import torch.nn as nn


class LinearQNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def save(self, file_path: str) -> None:
        torch.save(self.state_dict(), file_path)

    def load(self, file_path: str, map_location: str = "cpu") -> None:
        state = torch.load(file_path, map_location=map_location)
        self.load_state_dict(state)
