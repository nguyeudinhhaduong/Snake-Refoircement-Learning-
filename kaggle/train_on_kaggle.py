"""
Run this file on Kaggle Notebook/Script.

Steps on Kaggle:
1. Upload project files (or paste this script + snake_rl package).
2. Install dependencies: `pip install pygame numpy torch matplotlib`
3. Run: `python kaggle/train_on_kaggle.py --episodes 1200`
4. Download `/kaggle/working/models/snake_dqn.pth` and put it in local `models/`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is available when this runs in Kaggle.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from train import train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Snake DQN on Kaggle")
    parser.add_argument("--episodes", type=int, default=1200, help="Number of episodes")
    parser.add_argument(
        "--save",
        type=str,
        default="/kaggle/working/models/snake_dqn.pth",
        help="Output model path on Kaggle",
    )
    parser.add_argument("--render-every", type=int, default=0, help="Render every N episodes")
    parser.add_argument("--fps", type=int, default=30, help="FPS when rendering")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        episodes=args.episodes,
        save_path=args.save,
        render_every=args.render_every,
        fps=args.fps,
    )
