# 2048 Expectimax DLS Agent

An AI agent that plays **2048** using **depth-limited search** and the **expectimax algorithm**. The project compares several agent strategies, starting with simple greedy agents and ending with a search-based agent designed to handle the randomness of new tile placement.

## Overview

2048 is a stochastic puzzle game played on a 4x4 grid. After each move, a new tile appears in a random empty cell, so the agent cannot simply plan one guaranteed future path. To handle that uncertainty, this project uses **expectimax**, where player moves are treated as choice nodes and random tile spawns are treated as chance nodes.

The final agent searches ahead to a depth of 8 and evaluates possible future board states using a custom heuristic function.

## Agent Design

The agent evaluates board states using several factors:

- **Score**: rewards higher total game score
- **Filled cell penalty**: penalizes crowded boards, especially after the board is more than half full
- **Max tile value**: rewards creating larger tiles
- **Domino effect**: rewards rows or columns with steadily increasing/decreasing values
- **Smoothness**: penalizes large differences between neighboring tiles
- **Game-over penalty**: heavily penalizes terminal board states

The agent clones game states during search so it can simulate moves without changing the real board.

## Testing and Results

Each agent was tested across **100 simulation runs**.

Final DLS agent results:

- **Highest score:** 28,672
- **Average score:** 10,330.04
- **Standard deviation:** 5,631.97

The depth-limited expectimax agents strongly outperformed the random, reflex, and greedy agents. The main tradeoff was performance: deeper search produced better decisions, but also made heuristic tuning more sensitive and computationally expensive.

## What I Learned

This project helped me understand how search-based AI handles uncertain environments. Greedy strategies worked in simple cases, but they failed to plan around future randomness. Expectimax produced much stronger play because it considered both the agent’s choices and the probability of random tile spawns.

I also learned that heuristic design matters a lot. Small changes in weights could significantly affect performance, especially as search depth increased.

## Future Improvements

- Add adaptive search depth based on board complexity
- Dynamically adjust heuristic weights during gameplay
- Optimize performance to allow deeper search
- Experiment with reinforcement learning

## Tech Stack

- Python
- NumPy
- Search algorithms
- Heuristic evaluation
