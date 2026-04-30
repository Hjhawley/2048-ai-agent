import gymnasium as gym
import twenty_forty_eight

def get_valid_moves(env):
    valid_moves = []
    for action in range(env.action_space.n):
        temp_env = gym.make('TwentyFortyEight-v0')
        temp_env.unwrapped.state.board = env.unwrapped.state.board.copy()
        temp_env.unwrapped.state.score = env.unwrapped.state.score
        changed = temp_env.unwrapped.state.move(action)
        if changed:
            valid_moves.append(action)
        temp_env.close()
    return valid_moves

def agent_function(observation, env):
    """
    Really dumb reflex agent that swipes left if possible,
    then right, then up, then down
    """
    action_priority = [2, 3, 0, 1]  # Left, Right, Up, Down
    valid_moves = get_valid_moves(env)
    for action in action_priority:
        if action in valid_moves:
            return action
    return 1

def run_agent(render_mode=None):
    env = gym.make('TwentyFortyEight-v0', render_mode=render_mode)
    observation, info = env.reset()
    terminated = truncated = False
    if render_mode == "human":
        env.render()
    while not (terminated or truncated):
        action = agent_function(observation, env)
        observation, reward, terminated, truncated, info = env.step(action)
        if render_mode == "human":
            env.render()
    final_score = env.unwrapped.state.get_score()
    env.close()
    return final_score

def main():
    render_mode = "ansi"
    final_score = run_agent(render_mode=render_mode)
    print(f"Game over! Final score: {final_score}")

if __name__ == "__main__":
    main()
