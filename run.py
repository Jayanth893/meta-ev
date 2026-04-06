import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from tasks.easy import run_task as run_easy
from tasks.medium import run_task as run_medium
from tasks.hard import run_task as run_hard

def evaluate_all():
    print("=" * 50)
    print("  Evaluating Baseline Agent Across All Tasks")
    print("=" * 50 + "\n")

    total_easy = run_easy(50, "Easy")
    total_medium = run_medium(100, "Medium")
    total_hard = run_hard(200, "Hard")

    overall_reward = total_easy + total_medium + total_hard
    total_episodes = 50 + 100 + 200

    print("\n" + "=" * 50)
    print("  Final Performance Summary")
    print("=" * 50)
    print(f"  Easy   Reward  : {total_easy:.2f}  ({total_easy/50:.4f} avg)")
    print(f"  Medium Reward  : {total_medium:.2f}  ({total_medium/100:.4f} avg)")
    print(f"  Hard   Reward  : {total_hard:.2f}  ({total_hard/200:.4f} avg)")
    print(f"  Overall Total  : {overall_reward:.2f}")
    print(f"  Overall Mean   : {(overall_reward / total_episodes):.4f}")
    print("=" * 50)

if __name__ == "__main__":
    evaluate_all()
