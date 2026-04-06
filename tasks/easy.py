"""
tasks/easy.py
-------------
Easy task: 50 full episodes, max 10 steps each, using the standard
job pool and the BaselineAgent directly through the environment.

Candidate profile: Mid-level JS/React/Node developer.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env.environment import JobApplyEnv
from agents.baseline_agent import BaselineAgent


def run_task(episodes: int = 50, task_name: str = "Easy") -> float:
    """
    Run `episodes` multi-step episodes and return the cumulative reward.
    Each episode consists of env.max_steps job decisions (default 10).
    """
    env   = JobApplyEnv(max_steps=10)   # Standard 30-job pool loaded from JSON
    agent = BaselineAgent()
    total_reward = 0.0

    print(f"--- Running {task_name} Task ({episodes} episodes × {env.max_steps} steps) ---")

    for ep in range(episodes):
        state = env.reset()
        done  = False
        ep_reward = 0.0

        while not done:
            action = agent.act(state)
            state, reward, done, _ = env.step(action)
            ep_reward    += reward
            total_reward += reward

    mean = total_reward / episodes
    print(f"  Task      : {task_name}")
    print(f"  Episodes  : {episodes}")
    print(f"  Total Reward : {total_reward:.2f}")
    print(f"  Mean Reward  : {mean:.4f}")
    print("-" * 30)
    return total_reward


def evaluate(env: JobApplyEnv, agent, episodes: int = 200) -> float:
    """
    Shared evaluation helper — runs agent for `episodes` multi-step episodes
    and returns the mean per-episode reward.
    """
    total = 0.0
    for _ in range(episodes):
        state = env.reset()
        done  = False
        while not done:
            action = agent.act(state)
            state, reward, done, _ = env.step(action)
            total += reward
    return total / episodes


if __name__ == "__main__":
    run_task(50, "Easy")
