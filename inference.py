import requests
import random
import time

# Placeholder for OpenEnv agent inference logic
# This script is often used by the evaluator to run an agent against the server.

class SimpleAgent:
    def act(self, observation):
        # Very simple random baseline for validation
        return {
            "apply": random.choice([True, False]),
            "resume_version": random.choice(["general", "frontend_focused", "backend_focused", "fullstack"])
        }

def run_evaluation(server_url="http://localhost:7860"):
    agent = SimpleAgent()
    
    # Wait for server to be ready
    for _ in range(10):
        try:
            requests.get(f"{server_url}/health")
            break
        except:
            time.sleep(2)
            
    # Reset
    response = requests.post(f"{server_url}/reset")
    obs = response.json()
    
    done = False
    total_reward = 0
    while not done:
        action = agent.act(obs)
        response = requests.post(f"{server_url}/step", json=action)
        result = response.json()
        obs = result["observation"]
        total_reward += result["reward"]
        done = result["done"]
        
    print(f"Evaluation complete. Total Reward: {total_reward}")

def main():
    """Main function for inference logic."""
    pass

if __name__ == "__main__":
    main()
