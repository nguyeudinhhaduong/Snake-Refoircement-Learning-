from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from random import randint
from typing import List, Tuple

import numpy as np
import pygame


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


@dataclass(frozen=True)
class Point:
    x: int
    y: int


class SnakeEnv:
    def __init__(self, width: int = 640, height: int = 480, block_size: int = 20) -> None:
        self.width = width
        self.height = height
        self.block_size = block_size
        self.grid_width = self.width // self.block_size
        self.grid_height = self.height // self.block_size

        self.direction = Direction.RIGHT
        self.head = Point(self.width // 2, self.height // 2)
        self.snake: List[Point] = []
        self.score = 0
        self.food = Point(0, 0)
        self.frame_iteration = 0
        self._is_pygame_ready = False
        self._snake_surface = None
        self._head_surface = None
        self._food_surface = None
        self._bg_surface = None
        self._hud_surface = None

        self.reset()

    def reset(self) -> np.ndarray:
        self.direction = Direction.RIGHT
        self.head = Point(self.width // 2, self.height // 2)
        self.snake = [
            self.head,
            Point(self.head.x - self.block_size, self.head.y),
            Point(self.head.x - (2 * self.block_size), self.head.y),
        ]
        self.score = 0
        self.frame_iteration = 0
        self._place_food()
        return self.get_state()

    def _place_food(self) -> None:
        x = randint(0, self.grid_width - 1) * self.block_size
        y = randint(0, self.grid_height - 1) * self.block_size
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, int]:
        self.frame_iteration += 1
        old_distance = abs(self.head.x - self.food.x) + abs(self.head.y - self.food.y)
        food_before_step = self.food

        self._move(action)
        self.snake.insert(0, self.head)

        reward = 0.0
        game_over = False

        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10.0
            return self.get_state(), reward, game_over, self.score

        if self.head == self.food:
            self.score += 1
            reward = 10.0
            self._place_food()
        else:
            self.snake.pop()
            self.food = food_before_step
            new_distance = abs(self.head.x - self.food.x) + abs(self.head.y - self.food.y)
            reward = -0.08 if new_distance < old_distance else -0.18

        return self.get_state(), reward, game_over, self.score

    def is_collision(self, point: Point | None = None) -> bool:
        if point is None:
            point = self.head

        if point.x > self.width - self.block_size or point.x < 0 or point.y > self.height - self.block_size or point.y < 0:
            return True

        if point in self.snake[1:]:
            return True

        return False

    def get_state(self) -> np.ndarray:
        head = self.snake[0]
        point_l = Point(head.x - self.block_size, head.y)
        point_r = Point(head.x + self.block_size, head.y)
        point_u = Point(head.x, head.y - self.block_size)
        point_d = Point(head.x, head.y + self.block_size)

        dir_l = self.direction == Direction.LEFT
        dir_r = self.direction == Direction.RIGHT
        dir_u = self.direction == Direction.UP
        dir_d = self.direction == Direction.DOWN

        state = np.array(
            [
                (dir_r and self.is_collision(point_r)) or (dir_l and self.is_collision(point_l)) or (dir_u and self.is_collision(point_u)) or (dir_d and self.is_collision(point_d)),
                (dir_u and self.is_collision(point_r)) or (dir_d and self.is_collision(point_l)) or (dir_l and self.is_collision(point_u)) or (dir_r and self.is_collision(point_d)),
                (dir_d and self.is_collision(point_r)) or (dir_u and self.is_collision(point_l)) or (dir_r and self.is_collision(point_u)) or (dir_l and self.is_collision(point_d)),
                dir_l,
                dir_r,
                dir_u,
                dir_d,
                self.food.x < self.head.x,
                self.food.x > self.head.x,
                self.food.y < self.head.y,
                self.food.y > self.head.y,
            ],
            dtype=int,
        )

        return state

    def get_grid_state(self) -> np.ndarray:
        # Channels: 0=snake body, 1=head, 2=food, 3..6=heading one-hot (R, L, U, D).
        grid = np.zeros((7, self.grid_height, self.grid_width), dtype=np.float32)

        for segment in self.snake[1:]:
            sx = segment.x // self.block_size
            sy = segment.y // self.block_size
            if 0 <= sx < self.grid_width and 0 <= sy < self.grid_height:
                grid[0, sy, sx] = 1.0

        head_x = self.head.x // self.block_size
        head_y = self.head.y // self.block_size
        food_x = self.food.x // self.block_size
        food_y = self.food.y // self.block_size

        if 0 <= head_x < self.grid_width and 0 <= head_y < self.grid_height:
            grid[1, head_y, head_x] = 1.0
        if 0 <= food_x < self.grid_width and 0 <= food_y < self.grid_height:
            grid[2, food_y, food_x] = 1.0

        if self.direction == Direction.RIGHT:
            grid[3, :, :] = 1.0
        elif self.direction == Direction.LEFT:
            grid[4, :, :] = 1.0
        elif self.direction == Direction.UP:
            grid[5, :, :] = 1.0
        else:
            grid[6, :, :] = 1.0

        return grid

    def _move(self, action: np.ndarray) -> None:
        clockwise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clockwise.index(self.direction)

        if np.array_equal(action, np.array([1, 0, 0])):
            new_dir = clockwise[idx]
        elif np.array_equal(action, np.array([0, 1, 0])):
            new_dir = clockwise[(idx + 1) % 4]
        else:
            new_dir = clockwise[(idx - 1) % 4]

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += self.block_size
        elif self.direction == Direction.LEFT:
            x -= self.block_size
        elif self.direction == Direction.DOWN:
            y += self.block_size
        else:
            y -= self.block_size

        self.head = Point(x, y)

    def render(self, fps: int = 20, title: str = "Snake RL") -> bool:
        if not self._is_pygame_ready:
            pygame.init()
            self.display = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption(title)
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("segoeui", 22, bold=True)
            self.small_font = pygame.font.SysFont("segoeui", 16)
            self._build_surfaces()
            self._is_pygame_ready = True

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return False

        self.display.blit(self._bg_surface, (0, 0))

        for idx, segment in enumerate(self.snake):
            if idx == 0:
                self.display.blit(self._head_surface, (segment.x, segment.y))
            else:
                self.display.blit(self._snake_surface, (segment.x, segment.y))

        self.display.blit(self._food_surface, (self.food.x, self.food.y))

        self.display.blit(self._hud_surface, (8, 8))
        score_text = self.font.render(f"Score: {self.score}", True, (235, 240, 250))
        help_text = self.small_font.render("ESC de thoat", True, (165, 178, 196))
        self.display.blit(score_text, (18, 14))
        self.display.blit(help_text, (18, 42))

        pygame.display.flip()
        self.clock.tick(fps)
        return True

    def _build_surfaces(self) -> None:
        self._snake_surface = pygame.Surface((self.block_size, self.block_size), pygame.SRCALPHA)
        pygame.draw.rect(
            self._snake_surface,
            (46, 160, 120),
            pygame.Rect(0, 0, self.block_size, self.block_size),
            border_radius=5,
        )
        pygame.draw.rect(
            self._snake_surface,
            (16, 92, 68),
            pygame.Rect(4, 4, self.block_size - 8, self.block_size - 8),
            border_radius=4,
        )

        self._head_surface = pygame.Surface((self.block_size, self.block_size), pygame.SRCALPHA)
        pygame.draw.rect(
            self._head_surface,
            (98, 214, 158),
            pygame.Rect(0, 0, self.block_size, self.block_size),
            border_radius=6,
        )
        pygame.draw.circle(self._head_surface, (18, 28, 42), (7, 7), 2)
        pygame.draw.circle(self._head_surface, (18, 28, 42), (13, 7), 2)

        self._food_surface = pygame.Surface((self.block_size, self.block_size), pygame.SRCALPHA)
        pygame.draw.circle(
            self._food_surface,
            (252, 98, 98),
            (self.block_size // 2, self.block_size // 2),
            self.block_size // 2 - 1,
        )
        pygame.draw.circle(
            self._food_surface,
            (255, 198, 198),
            (self.block_size // 2 - 4, self.block_size // 2 - 4),
            3,
        )

        self._bg_surface = pygame.Surface((self.width, self.height))
        top = np.array([10, 17, 28], dtype=np.float32)
        bottom = np.array([22, 32, 48], dtype=np.float32)
        for y in range(self.height):
            blend = y / max(1, self.height - 1)
            color = (top * (1.0 - blend) + bottom * blend).astype(np.uint8)
            pygame.draw.line(self._bg_surface, color.tolist(), (0, y), (self.width, y))

        grid_color = (32, 50, 74)
        for x in range(0, self.width, self.block_size):
            pygame.draw.line(self._bg_surface, grid_color, (x, 0), (x, self.height), 1)
        for y in range(0, self.height, self.block_size):
            pygame.draw.line(self._bg_surface, grid_color, (0, y), (self.width, y), 1)

        self._hud_surface = pygame.Surface((190, 64), pygame.SRCALPHA)
        pygame.draw.rect(self._hud_surface, (9, 13, 22, 170), pygame.Rect(0, 0, 190, 64), border_radius=10)

    def close(self) -> None:
        if self._is_pygame_ready:
            pygame.quit()
            self._is_pygame_ready = False
