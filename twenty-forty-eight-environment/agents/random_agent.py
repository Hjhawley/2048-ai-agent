import gymnasium as gym
import twenty_forty_eight
import random

def agent_function(observation):
    return random.choice([0, 1, 2, 3])

def run_agent(render_mode=None):
    env = gym.make('TwentyFortyEight-v0', render_mode=render_mode)
    observation, info = env.reset()
    terminated = truncated = False
    while not (terminated or truncated):
        action = agent_function(observation)
        observation, reward, terminated, truncated, info = env.step(action)
    final_score = env.unwrapped.state.get_score()
    env.close()
    return final_score

def main():
    render_mode = "ansi"
    final_score = run_agent(render_mode=render_mode)
    print(f"Game over! Final score: {final_score}")

if __name__ == "__main__":
    main()
