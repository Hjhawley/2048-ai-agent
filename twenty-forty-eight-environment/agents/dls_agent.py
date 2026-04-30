import gymnasium as gym
import twenty_forty_eight
import numpy as np
import copy

# Expectimax with depth-limited search

def heuristic(board, score, is_terminal):
    """
    - Penalizes the number of filled cells exponentially
    - Max tile value (push for 2048)
    - Domino effect (rows and columns in decreasing order)
    - Smoothness (penalize differences between neighboring tiles)
    - Increased score
    """
    total_cells = board.size  # Total number of cells, e.g., 16 for 4x4 board
    filled_cells = np.count_nonzero(board != 0)
    
    max_tile = np.max(board)
    domino = calculate_domino(board)
    smoothness = calculate_smoothness(board)

    # Fine-tune these weights
    filled_penalty_weight = 0.05
    max_tile_weight = 1.5
    domino_weight = 0.1
    smoothness_weight = 0.1
    score_weight = 0.025

    # Define the threshold after which penalty grows exponentially
    threshold = total_cells // 2  # Half-filled board
    if filled_cells > threshold:
        penalty_exponent = filled_cells - threshold
        filled_penalty = filled_penalty_weight * (2 ** penalty_exponent)
    else:
        filled_penalty = 0

    # Total heuristic score
    total_score = (max_tile_weight * np.log2(max_tile) +
                   domino_weight * domino -
                   smoothness_weight * smoothness -
                   filled_penalty +
                   score_weight)

    # Subtract penalty if the state is terminal (game over)
    if is_terminal:
        total_score -= 100  # Game over penalty

    return total_score

def calculate_domino(board):
    # "domino effect" how consistently tile values increase/decrease in a row
    totals = [0, 0, 0, 0]

    # Rows
    for row in board:
        for i in range(len(row) - 1):
            if row[i] > row[i + 1]:
                totals[0] += row[i] - row[i + 1]
            elif row[i] < row[i + 1]:
                totals[1] += row[i + 1] - row[i]

    # Columns
    for col in board.T:
        for i in range(len(col) - 1):
            if col[i] > col[i + 1]:
                totals[2] += col[i] - col[i + 1]
            elif col[i] < col[i + 1]:
                totals[3] += col[i + 1] - col[i]

    return -min(totals[0], totals[1]) - min(totals[2], totals[3])

def calculate_smoothness(board):
    # penalizes differences between neighboring tiles
    smoothness = 0
    for x in range(board.shape[0]):
        for y in range(board.shape[1]):
            value = board[x][y]
            if value != 0:
                value = np.log2(value)
                for direction in [(1, 0), (0, 1)]:
                    dx = x + direction[0]
                    dy = y + direction[1]
                    if 0 <= dx < board.shape[0] and 0 <= dy < board.shape[1]:
                        target = board[dx][dy]
                        if target != 0:
                            target = np.log2(target)
                            smoothness -= abs(value - target)
    return smoothness

def clone_env_state(original_env):
    # Create a new environment instance
    new_env = gym.make('TwentyFortyEight-v0')
    # Manually copy the board and score
    new_env.unwrapped.state.board = original_env.unwrapped.state.board.copy()
    new_env.unwrapped.state.score = original_env.unwrapped.state.score
    return new_env

def get_valid_moves(env):
    valid_moves = []
    for action in range(env.action_space.n):
        temp_env = clone_env_state(env)
        changed = temp_env.unwrapped.state.move(action)
        if changed:
            valid_moves.append(action)
        temp_env.close()
    return valid_moves

def expectimax(env, depth, is_player_turn=True):
    if depth == 0:
        return heuristic(env.unwrapped.state.board, env.unwrapped.state.score, is_terminal=False)
    
    if is_player_turn: # Choice nodes
        valid_moves = get_valid_moves(env)
        if not valid_moves: # Terminal state
            return heuristic(env.unwrapped.state.board, env.unwrapped.state.score, is_terminal=True)
        max_utility = -float('inf')
        for action in valid_moves:
            temp_env = clone_env_state(env)
            temp_env.unwrapped.state.move(action)
            utility = expectimax(temp_env, depth - 1, False)
            max_utility = max(max_utility, utility)
            temp_env.close()
        return max_utility
    else: # Chance nodes
        empty_cells = list(zip(*np.where(env.unwrapped.state.board == 0)))
        if not empty_cells:
            valid_moves = get_valid_moves(env)
            if not valid_moves:
                return heuristic(env.unwrapped.state.board, env.unwrapped.state.score, is_terminal=True)
            else:
                # Proceed to player's turn if moves are available
                return expectimax(env, depth - 1, True)
        
        utilities = []
        for cell in empty_cells:
            for tile, prob in [(2, 0.5), (4, 0.5)]:
                temp_env = clone_env_state(env)
                temp_env.unwrapped.state.board[cell] = tile
                utility = expectimax(temp_env, depth - 1, True)
                utilities.append(utility * prob)
                temp_env.close()
        # Average over all possible additions
        return sum(utilities) / len(utilities)

def agent_function(observation, env, depth=3):
    best_score = -float('inf')
    best_action = None
    valid_moves = get_valid_moves(env)
    if not valid_moves:
        return 0  # Default action if no valid moves
    #current_score = env.unwrapped.state.score  # Retrieve current score
    for action in valid_moves:
        temp_env = clone_env_state(env)
        temp_env.unwrapped.state.move(action)
        score = expectimax(temp_env, depth - 1, False)
        if score > best_score:
            best_score = score
            best_action = action
        temp_env.close()
    return best_action

def run_agent(render_mode=None, depth=3):
    env = gym.make('TwentyFortyEight-v0', render_mode=render_mode)
    observation, info = env.reset()
    terminated = truncated = False
    while not (terminated or truncated):
        action = agent_function(observation, env, depth=depth)
        observation, reward, terminated, truncated, info = env.step(action)
        if render_mode == "human":
            env.render()
    final_score = env.unwrapped.state.get_score()
    env.close()
    return final_score

def main():
    render_mode = "human"
    depth = 8  # fine tune depth limit
    final_score = run_agent(render_mode=render_mode, depth=depth)
    print(f"Game over! Final score: {final_score}")

if __name__ == "__main__":
    main()
