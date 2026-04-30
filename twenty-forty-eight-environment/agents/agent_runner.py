import sys
import argparse
import importlib
import statistics

"""
Ex:
python agent_runner.py random_agent 3 --render
python agent_runner.py greedy_agent 10
python agent_runner.py dls_agent 100
"""

def main():
    parser = argparse.ArgumentParser(description='Run an agent multiple times and average the final scores.')
    parser.add_argument('agent', type=str, help='The agent to run (e.g., random_agent or search_agent)')
    parser.add_argument('runs', type=int, help='The number of times to run the agent')
    parser.add_argument('--render', action='store_true', help='Render the game graphically')
    args = parser.parse_args()

    agent_name = args.agent
    num_runs = args.runs
    render_mode = 'human' if args.render else None

    try:
        agent_module = importlib.import_module(agent_name)
    except ImportError:
        print(f"Don't forget pip install -e .")
        sys.exit(1)

    scores = []
    best_score = 0
    for i in range(num_runs):
        if render_mode == 'human' and num_runs > 1:
            print("WARNING: Running multiple games in human mode")
        final_score = agent_module.run_agent(render_mode=render_mode)
        final_score = int(final_score)
        scores.append(final_score)
        if final_score > best_score:
            best_score = final_score
        print(f"Run {i+1}: Final score = {final_score}")

    average_score = sum(scores) / len(scores)
    std_dev = statistics.stdev(scores) if len(scores) > 1 else 0

    print(f"\nAgent: {agent_name}")
    print(f"Number of runs: {num_runs}")
    print(f"Highest score: {best_score}")
    print(f"Average final score: {average_score:.2f}")
    print(f"Standard deviation: {std_dev:.2f}")

if __name__ == "__main__":
    main()
