from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from snake_rl.agent import DQNAgent
from snake_rl.env import SnakeEnv


def train(
    episodes: int,
    save_path: str,
    render_every: int = 0,
    fps: int = 30,
    hidden_size: int = 384,
) -> None:
    env = SnakeEnv()
    agent = DQNAgent(hidden_size=hidden_size)

    scores = []
    mean_scores = []
    total_score = 0
    record = 0

    save_file = Path(save_path)
    save_file.parent.mkdir(parents=True, exist_ok=True)

    for episode in range(1, episodes + 1):
        state = env.reset()
        done = False

        while not done:
            action = agent.get_action(state, train=True)
            next_state, reward, done, score = env.step(action)

            agent.remember(state, action, reward, next_state, done)
            agent.train_from_replay()

            state = next_state

            if render_every > 0 and episode % render_every == 0:
                keep_running = env.render(fps=fps, title=f"Training - Episode {episode}")
                if not keep_running:
                    done = True
                    break

        agent.train_long_memory()
        agent.update_epsilon()

        if score > record:
            record = score
            agent.save(str(save_file))

        scores.append(score)
        total_score += score
        mean_score = total_score / episode
        mean_scores.append(mean_score)

        if episode % 20 == 0 or episode == 1:
            print(
                f"Episode {episode}/{episodes} | Score: {score} | Record: {record} | "
                f"Mean: {mean_score:.2f} | Epsilon: {agent.epsilon:.4f}"
            )

    agent.save(str(save_file.with_name("snake_dqn_last.pth")))
    env.close()

    _save_training_plot(scores, mean_scores, save_file.with_name("training_plot.png"))
    print(f"Training completed. Best model saved at: {save_file}")


def _save_training_plot(scores, mean_scores, output_path: Path) -> None:
    plt.figure(figsize=(10, 5))
    plt.title("Snake DQN Training")
    plt.xlabel("Games")
    plt.ylabel("Score")
    plt.plot(scores, label="Score per Game", alpha=0.7)
    plt.plot(mean_scores, label="Mean Score", linewidth=2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Snake DQN Agent")
    parser.add_argument("--episodes", type=int, default=500, help="Number of games to train")
    parser.add_argument("--save", type=str, default="models/snake_dqn.pth", help="Path to save the best model")
    parser.add_argument("--render-every", type=int, default=0, help="Render training every N episodes (0 = no render)")
    parser.add_argument("--fps", type=int, default=30, help="FPS when rendering during training")
    parser.add_argument("--hidden-size", type=int, default=384, help="Hidden layer size of Q-network")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        episodes=args.episodes,
        save_path=args.save,
        render_every=args.render_every,
        fps=args.fps,
        hidden_size=args.hidden_size,
    )
