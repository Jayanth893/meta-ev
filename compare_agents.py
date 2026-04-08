"""
compare_agents.py
-----------------
Trains the Q-Learning agent against the multi-step environment and then
benchmarks it against the rule-based BaselineAgent over a large evaluation run.

Usage:
    py compare_agents.py
"""

import sys
import os

# Ensure the project root is on sys.path regardless of where the script is run from.
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from env.environment import JobApplyEnv
from agents.baseline_agent import BaselineAgent
from agents.rl_agent import QLearningAgent


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def evaluate_agent(env: JobApplyEnv, agent, episodes: int = 1000) -> float:
    """
    Run `episodes` full multi-step episodes and return the mean per-episode reward.

    Each episode contains env.max_steps job decisions.  The total reward is the
    sum across all steps; we then divide by the number of episodes (not steps)
    so the number is comparable across different max_steps settings.
    """
    total_reward = 0.0
    for _ in range(episodes):
        state = env.reset()
        done = False
        while not done:
            action = agent.act(state)          # Baseline and RL share the same `act` signature
            state, reward, done, _ = env.step(action)
            total_reward += reward
    return total_reward / episodes


# ---------------------------------------------------------------------------
# Training + comparison entry-point
# ---------------------------------------------------------------------------

def train_and_compare():
    # Shared environment — max_steps=10 means 10 job decisions per episode.
    env = JobApplyEnv(max_steps=10, enable_logging=False)
    baseline = BaselineAgent()
    rl_agent  = QLearningAgent(alpha=0.1, gamma=0.9, epsilon=0.15)

    print("=" * 50)
    print("  Phase 1: Training RL Agent (multi-step)")
    print("=" * 50)

    train_episodes = 20_000
    for ep in range(train_episodes):
        state = env.reset()
        done  = False
        while not done:
            # Act with exploration enabled
            action = rl_agent.act(state, training=True)
            next_state, reward, done, _ = env.step(action)
            # Update Q-table: pass next_state so Bellman can bootstrap correctly
            rl_agent.update(reward, next_state_env=next_state, done=done)
            state = next_state  # Advance state

    print(f"  Training complete — {train_episodes} episodes")
    print(f"  Unique (state, action) pairs in Q-table: {len(rl_agent.q_table)}")
    
    # Save the trained model
    rl_agent.save_model(os.path.join(os.path.dirname(__file__), "rl_model.json"))
    print(f"  Model saved to 'rl_model.json'")

    print("\n" + "=" * 50)
    print("  Phase 2: Evaluation (10 000 episodes each)")
    print("=" * 50)

    eval_episodes = 10_000
    base_mean = evaluate_agent(env, baseline, eval_episodes)
    rl_mean   = evaluate_agent(env, rl_agent,  eval_episodes)

    print(f"  Baseline Agent  — Mean episode reward: {base_mean:+.4f}")
    print(f"  RL (Q-Learning) — Mean episode reward: {rl_mean:+.4f}")
    print("-" * 50)

    diff = rl_mean - base_mean
    if diff > 0:
        print(f"  ✅ RL Agent outperformed Baseline by {diff:.4f}")
    elif diff < 0:
        print(f"  📊 Baseline outperformed RL Agent by {-diff:.4f}")
    else:
        print("  🔁 Both agents performed equally well.")
    print("=" * 50)


if __name__ == "__main__":
    train_and_compare()
