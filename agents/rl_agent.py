import random
import os
from typing import Dict, Any, Tuple

class QLearningAgent:
    def __init__(self, alpha: float = 0.1, gamma: float = 0.9, epsilon: float = 0.1):
        # Q-table: mapping from (state, action) -> q_value
        self.q_table = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
        self.resume_versions = ["general", "frontend_focused", "backend_focused", "fullstack"]
        # Actions space: apply (bool) and resume_version (str)
        self.actions = [(True, rv) for rv in self.resume_versions] + [(False, rv) for rv in self.resume_versions]

    def _extract_state(self, state: Dict[str, Any]) -> Tuple:
        """Converts the environment dictionary state into a discrete hashable tuple."""
        cand_skills = [s.lower() for s in state.get("candidate_skills", [])]
        req_skills = [s.lower() for s in state.get("required_skills", [])]
        
        cand_exp = state.get("experience_level", "mid").lower()
        req_exp = state.get("experience_required", "not specified").lower()
        
        # Skill matching: How many candidate skills are actually required?
        skill_match_count = sum(1 for skill in cand_skills if skill in req_skills)
        
        # Location matching
        job_loc = state.get("location", "remote").lower()
        is_remote = 1 if "remote" in job_loc else 0
        
        # Salary level categorization
        sal_str = state.get("salary", "0").lower()
        if any(x in sal_str for x in ["150", "160", "170", "180", "190", "200"]):
            sal_level = "high"
        elif any(x in sal_str for x in ["100", "110", "120", "130", "140"]):
            sal_level = "mid"
        else:
            sal_level = "low"

        # Job Type classification (Frontend / Backend / Fullstack)
        job_desc = state.get("job_description", "").lower()
        front_kws = ["front", "react", "ui", "ux", "vue", "web"]
        back_kws = ["back", "node", "api", "server", "docker", "cloud", "aws", "data", "ml"]
        
        is_front = any(k in job_desc for k in front_kws)
        is_back = any(k in job_desc for k in back_kws)
        
        if is_front and is_back:
            job_type = "fullstack"
        elif is_front:
            job_type = "frontend"
        elif is_back:
            job_type = "backend"
        else:
            job_type = "other"
            
        return (skill_match_count, cand_exp, req_exp, is_remote, sal_level, job_type)

    def _get_q(self, state: Tuple, action: Tuple) -> float:
        return self.q_table.get((state, action), 0.0)

    def act(self, env_state: Dict[str, Any], training: bool = False) -> Dict[str, Any]:
        state = self._extract_state(env_state)
        
        # Epsilon-greedy exploration
        if training and random.uniform(0, 1) < self.epsilon:
            # Explore
            action = random.choice(self.actions)
        else:
            # Exploit
            q_values = [self._get_q(state, a) for a in self.actions]
            max_q = max(q_values)
            # Find all actions with the maximum Q-value to break ties randomly
            best_actions = [a for a, q in zip(self.actions, q_values) if q == max_q]
            action = random.choice(best_actions)
            
        # Store for update step
        self.last_state = state
        self.last_action = action
        
        return {"apply": action[0], "resume_version": action[1]}

    def update(self, reward: float, next_state_env: Dict[str, Any] = None, done: bool = True):
        if hasattr(self, 'last_state') and hasattr(self, 'last_action'):
            old_q = self._get_q(self.last_state, self.last_action)
            
            if done or next_state_env is None:
                max_next_q = 0.0
            else:
                next_state = self._extract_state(next_state_env)
                max_next_q = max([self._get_q(next_state, a) for a in self.actions])
                
            # Bellman update
            new_q = old_q + self.alpha * (reward + self.gamma * max_next_q - old_q)
            self.q_table[(self.last_state, self.last_action)] = new_q

    def feedback(self, state: Dict[str, Any], reward: float):
        """Allows the agent to learn from the reward of its last action."""
        # Note: This is a simplified feedback for online learning during app interaction.
        # It assumes the environment has already transitioned, but doesn't strictly 
        # require the next state for this simplified update.
        self.update(reward, next_state_env=None, done=True)
    def save_model(self, filepath: str):
        """Saves current Q-table to a JSON file."""
        import json
        # Convert tuple keys to strings for JSON
        serializable_q_table = {str(k): v for k, v in self.q_table.items()}
        with open(filepath, "w") as f:
            json.dump(serializable_q_table, f)

    def load_model(self, filepath: str):
        """Loads Q-table from a JSON file."""
        import json
        import ast
        if not os.path.exists(filepath):
            return
        with open(filepath, "r") as f:
            data = json.load(f)
            # Convert string keys back to tuples using ast.literal_eval
            self.q_table = {ast.literal_eval(k): v for k, v in data.items()}
