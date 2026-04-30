import numpy as np
import random

class TwentyFortyEightState:

    def __init__(self, size=4):
        self.size = size
        self.board = np.zeros((self.size, self.size), dtype=np.int32)
        self.score = 0
        self.last_move_score = 0

    def reset(self):
        self.board = np.zeros((self.size, self.size), dtype=np.int32)
        self.score = 0
        self.last_move_score = 0
        self.add_random_tile()
        self.add_random_tile()

    def add_random_tile(self):
        """
        Adds a 2 or 4 to an empty cell on the board
        """
        empty_cells = list(zip(*np.where(self.board == 0)))
        if empty_cells:
            row, col = random.choice(empty_cells)
            self.board[row, col] = random.choice([2, 4])

    def move(self, direction):
        """
        0=up, 1=down, 2=left, 3=right
        Return True if the board changed, else False
        """
        self.last_move_score = 0
        old_board = self.board.copy()
        if direction == 0:
            self.board = self.move_up(self.board)
        elif direction == 1:
            self.board = self.move_down(self.board)
        elif direction == 2:
            self.board = self.move_left(self.board)
        elif direction == 3:
            self.board = self.move_right(self.board)
        else:
            raise ValueError("Not a valid direction")
        changed = not np.array_equal(old_board, self.board)
        if changed:
            self.add_random_tile()
        return changed

    def move_left(self, board):
        new_board = np.zeros_like(board)
        for row in range(self.size):
            tiles = board[row, board[row, :] != 0]
            new_row, score_gain = self.merge_tiles(tiles)
            self.last_move_score += score_gain
            new_board[row, :len(new_row)] = new_row
        self.score += self.last_move_score
        return new_board

    def move_right(self, board):
        new_board = np.zeros_like(board)
        for row in range(self.size):
            tiles = board[row, board[row, :] != 0]
            tiles = tiles[::-1]
            new_row, score_gain = self.merge_tiles(tiles)
            self.last_move_score += score_gain
            new_row = new_row[::-1]
            start_idx = self.size - len(new_row)
            if len(new_row) > 0:
                new_board[row, start_idx:] = new_row
        self.score += self.last_move_score
        return new_board

    def move_up(self, board):
        board = board.T
        new_board = self.move_left(board)
        return new_board.T

    def move_down(self, board):
        board = board.T
        new_board = self.move_right(board)
        return new_board.T

    def merge_tiles(self, tiles):
        """
        Merges the tiles in a row or column, returns new tiles and points scored
        """
        new_tiles = []
        score_gain = 0
        skip = False
        i = 0
        while i < len(tiles):
            if skip:
                skip = False
                i += 1
                continue
            if i + 1 < len(tiles) and tiles[i] == tiles[i + 1]:
                merged_value = tiles[i] * 2
                score_gain += merged_value
                new_tiles.append(merged_value)
                skip = True
            else:
                new_tiles.append(tiles[i])
            i += 1
        return np.array(new_tiles, dtype=np.int32), score_gain

    def is_game_over(self):
        for direction in range(4):
            temp_board = self.board.copy()
            temp_state = TwentyFortyEightState(self.size)
            temp_state.board = temp_board
            if temp_state.move(direction):
                return False
        return True

    def get_observation(self):
        return self.board.copy()

    def get_score(self):
        return self.score

    def get_last_move_score(self):
        return self.last_move_score

    def __str__(self):
        """
        String representation of the board
        """
        return '\n'.join(['\t'.join(map(str, row)) for row in self.board])

if __name__ == "__main__":
    state = TwentyFortyEightState()
    state.reset()
    print(state)
    state.move(0)  # Up
    print(state)
    state.move(1)  # Down
    print(state)
    state.move(2)  # Left
    print(state)
    state.move(3)  # Right
    print(state)
