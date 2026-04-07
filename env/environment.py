import random
import json
import os
import re
from datetime import datetime
from typing import Dict, Any, Tuple
from env.models import State

class JobApplyEnv:
    def __init__(self, max_steps=10):
        # Load jobs from JSON file
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(base_dir, "data", "jobs.json"), "r") as f:
            self.jobs = json.load(f)

        self.candidate = {
            "skills": ["JavaScript", "React", "Node.js", "MongoDB"],
            "experience": "Mid",
            "preferred_location": "Remote",
            "salary_expectation": 110000
        }
        
        self.max_steps = max_steps
        self.current_step = 0
        self.cumulative_reward = 0.0
        self.current_job = None
        self.reset()

    def reset(self):
        self.current_step = 0
        self.cumulative_reward = 0.0
        self.current_job = random.choice(self.jobs)
        return self.state()

    def state(self):
        return State(
            self.candidate["skills"],
            self.candidate["experience"],
            self.current_job.get("description", ""),
            self.current_job.get("required_skills", []),
            self.current_job.get("company_type", "startup"),
            self.current_job.get("experience_level", "Mid"),
            self.current_job.get("location", "Remote"),
            self.current_job.get("salary", "Competitive"),
            current_step=self.current_step,
            max_steps=self.max_steps,
            cumulative_reward=self.cumulative_reward
        ).to_dict()

    def _parse_salary(self, salary_str: str) -> Tuple[int, int]:
        """Simple parser to convert string like '$70k - $90k' to numeric min/max."""
        try:
            # Handle cases like "90k - 120k GBP" or "$70k - $90k"
            nums = re.findall(r'(\d+)', salary_str.lower())
            if not nums: return 0, 0
            
            # Convert 'k' to 1000s
            mult = 1000 if 'k' in salary_str.lower() else 1
            if len(nums) >= 2:
                return int(nums[0]) * mult, int(nums[1]) * mult
            elif len(nums) == 1:
                return int(nums[0]) * mult, int(nums[0]) * mult
        except:
            pass
        return 0, 0

    def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict]:
        # Capture current state before advancing
        current_state_dict = self.state()
        
        reward = self.calculate_reward(action)
        self.current_step += 1
        self.cumulative_reward += reward
        
        done = self.current_step >= self.max_steps
        
        # Log the interaction
        self._log_step(current_state_dict, action, reward)
        
        if not done:
            self.current_job = random.choice(self.jobs)
            
        return self.state(), reward, done, {}

    def _log_step(self, state: Dict[str, Any], action: Dict[str, Any], reward: float):
        """Append a structured log entry of the step to logs.txt."""
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        log_file = os.path.join(base_dir, "logs.txt")
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Format the log block
        log_entry = (
            f"[{timestamp}] STEP {state.get('current_step', 0)} "
            f"(Cumulative Reward: {state.get('cumulative_reward', 0.0):.2f})\n"
            f"  [STATE] Job: {state.get('company_type')} | Reqs: {state.get('experience_required')} | Loc: {state.get('location')} | Sal: {state.get('salary')}\n"
            f"  [ACTION] Apply: {action.get('apply')} | Resume: {action.get('resume_version')}\n"
            f"  [REWARD] {reward:+.2f}\n"
            f"{'-'*50}\n"
        )
        
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(log_entry)

    def calculate_reward(self, action: Dict[str, Any]) -> float:
        # Core data
        cand_skills = set(s.lower() for s in self.candidate["skills"])
        req_skills = set(s.lower() for s in self.current_job.get("required_skills", []))
        
        # 1. Skill Match (0.0 to 0.6)
        skill_score = 0
        if req_skills:
            intersection = cand_skills.intersection(req_skills)
            skill_score = (len(intersection) / len(req_skills)) * 0.6
        
        # 2. Experience Alignment (-0.2 to +0.1)
        exp_levels = {"junior": 1, "mid": 2, "senior": 3, "not specified": 2}
        req_exp = exp_levels.get(self.current_job.get("experience_level", "mid").lower(), 2)
        cand_exp = exp_levels.get(self.candidate["experience"].lower(), 2)
        
        exp_diff = cand_exp - req_exp
        exp_impact = 0
        if exp_diff < 0: exp_impact = -0.2
        elif exp_diff == 0: exp_impact = 0.1
        
        # 3. Location Match (0.0 or 0.1)
        job_loc = self.current_job.get("location", "Remote").lower()
        pref_loc = self.candidate["preferred_location"].lower()
        loc_impact = 0.1 if (job_loc == "remote" or pref_loc in job_loc) else -0.1
        
        # 4. Salary Alignment (-0.3 to +0.2)
        salary_str = self.current_job.get("salary", "0")
        min_sal, max_sal = self._parse_salary(salary_str)
        expectation = self.candidate["salary_expectation"]
        
        sal_impact = 0
        if max_sal > 0:
            if max_sal < expectation * 0.8:
                sal_impact = -0.3 # Significantly below expectation
            elif min_sal >= expectation:
                sal_impact = 0.2 # Exceeds expectation
            else:
                sal_impact = 0.1 # Within range
                
        # Aggregate Quality (Ideal range roughly -0.5 to 1.0)
        match_quality = skill_score + exp_impact + loc_impact + sal_impact
        
        if action["apply"]:
            # Final Reward clamped between -1 and 1
            return round(max(-1.0, min(1.0, match_quality)), 2)
        else:
            # Smart skip reward
            if match_quality < 0.2:
                return 0.2 # Good skip
            elif match_quality > 0.7:
                return -0.4 # Missed opportunity
            else:
                return 0.0
