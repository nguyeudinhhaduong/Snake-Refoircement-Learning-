from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pygame
import torch

from snake_rl.agent import DQNAgent
from snake_rl.env import SnakeEnv


def _infer_model_dims(model_path: Path) -> tuple[int, int, int]:
    state = torch.load(str(model_path), map_location="cpu")
    input_size = int(state["model.0.weight"].shape[1])
    hidden_size = int(state["model.0.weight"].shape[0])
    output_size = int(state["model.4.weight"].shape[0])
    return input_size, hidden_size, output_size


def play_ai(model_path: str, fps: int) -> None:
    model_file = Path(model_path)
    if not model_file.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_file}. Train first or copy model into models/."
        )

    env = SnakeEnv()
    state_size, hidden_size, action_size = _infer_model_dims(model_file)
    agent = DQNAgent(state_size=state_size, action_size=action_size, hidden_size=hidden_size)
    agent.load(str(model_file))

    running = True
    while running:
        state = env.get_state()
        action = agent.get_action(state, train=False)
        _, _, done, _ = env.step(action)
        running = env.render(fps=fps, title="Snake RL - AI")

        if done:
            env.reset()

    env.close()


def play_human(fps: int) -> None:
    env = SnakeEnv()

    action = np.array([1, 0, 0])
    running = env.render(fps=fps, title="Snake RL - Human")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RIGHT:
                    action = np.array([0, 1, 0])
                elif event.key == pygame.K_LEFT:
                    action = np.array([0, 0, 1])
                elif event.key == pygame.K_UP:
                    action = np.array([1, 0, 0])

        _, _, done, _ = env.step(action)
        running = running and env.render(fps=fps, title="Snake RL - Human")

        # Keep the latest turn command for one frame so controls feel responsive.
        action = np.array([1, 0, 0])

        if done:
            env.reset()

    env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play Snake (AI or Human)")
    parser.add_argument("--mode", type=str, default="ai", choices=["ai", "human"], help="Play mode")
    parser.add_argument("--model", type=str, default="models/snake_dqn.pth", help="Path to model file")
    parser.add_argument("--fps", type=int, default=45, help="Game FPS")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "ai":
        play_ai(model_path=args.model, fps=args.fps)
    else:
        play_human(fps=args.fps)
