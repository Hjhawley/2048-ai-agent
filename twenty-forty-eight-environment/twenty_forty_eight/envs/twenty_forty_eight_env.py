import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces
from twenty_forty_eight.envs.twenty_forty_eight_model import TwentyFortyEightState

class TwentyFortyEightEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array", "ansi"],
        "render_fps": 24,
    }

    def __init__(self, render_mode=None):
        self.render_mode = render_mode
        self.size = 4  # 4x4 board
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0, high=2 ** 16, shape=(self.size, self.size), dtype=np.int32
        )
        self.state = TwentyFortyEightState(self.size)

        # For rendering
        self.window_width = 512
        self.score_height = 64  # space reserved for score display
        self.window_height = self.window_width + self.score_height
        self.window = None
        self.clock = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state.reset()
        observation = self.state.get_observation()
        info = {'score': self.state.get_score()}
        if self.render_mode == "human":
            self.render()
        return observation, info

    def step(self, action):
        assert self.action_space.contains(action), "Invalid action"
        changed = self.state.move(action)
        observation = self.state.get_observation()
        reward = self.state.get_last_move_score()
        terminated = self.state.is_game_over()
        info = {'score': self.state.get_score()}
        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "ansi":
            board_str = '\n'.join(['\t'.join(map(str, row)) for row in self.state.board])
            return board_str
        elif self.render_mode == "human":
            self._render_gui()
        elif self.render_mode == "rgb_array":
            return self._render_gui()

    def _render_gui(self):
        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((self.window_width, self.window_height))
            pygame.display.set_caption("2048")
        if self.clock is None:
            self.clock = pygame.time.Clock()
        self.window.fill((0, 95, 95))  # Background color

        font = pygame.font.Font(None, 36)
        score_text = font.render(f"Score: {self.state.get_score()}", True, (255, 255, 255))
        score_rect = score_text.get_rect(center=(self.window_width / 2, self.score_height / 2))
        self.window.blit(score_text, score_rect)

        tile_size = self.window_width / self.size
        for row in range(self.size):
            for col in range(self.size):
                value = self.state.board[row, col]
                tile_rect = pygame.Rect(
                    col * tile_size,
                    self.score_height + row * tile_size,  # Adjust for score display height
                    tile_size,
                    tile_size
                )
                pygame.draw.rect(
                    self.window,
                    self._get_tile_color(value),
                    tile_rect,
                    border_radius=8,
                )
                if value != 0:
                    font = pygame.font.Font(None, 40)
                    text = font.render(str(value), True, (119, 110, 101))
                    text_rect = text.get_rect(center=tile_rect.center)
                    self.window.blit(text, text_rect)

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def _get_tile_color(self, value):
        tile_colors = {
            0: (205, 195, 180),
            2: (235, 225, 215),
            4: (235, 225, 200),
            8: (240, 175, 120),
            16: (245, 150, 100),
            32: (245, 125, 95),
            64: (245, 95, 60),
            128: (235, 205, 115),
            256: (235, 205, 95),
            512: (235, 200, 80),
            1024: (235, 195, 65),
            2048: (235, 195, 45),
        }
        return tile_colors.get(value, (255, 255, 255))  # after 2048, use some default color

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
