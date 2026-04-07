from dataclasses import dataclass, field
from typing import List, Literal, Dict

@dataclass
class CandidateProfile:
    """Represents a job candidate looking for positions."""
    skills: List[str]
    experience_level: Literal["Junior", "Mid", "Senior"]

@dataclass
class JobProfile:
    """Represents a single job opportunity in the simulation."""
    title: str
    description: str
    required_skills: List[str]
    company_type: Literal["Startup", "MNC", "SME"]
    experience_required: Literal["Junior", "Mid", "Senior"]
    location: str
    salary: str

@dataclass
class Action:
    """Represents a decision taken by an agent for a job."""
    apply: bool
    resume_version: Literal["frontend_focused", "backend_focused", "fullstack", "general"]

@dataclass
class RewardInfo:
    """Represents the reward breakdown after step completion."""
    reward: float
    message: str

class State:
    """
    Captures the current state of the environment, including the candidate's profile,
    the current job details, and episode progression metrics.
    """
    def __init__(self, 
                 candidate_skills: List[str], 
                 experience_level: str, 
                 job_description: str, 
                 required_skills: List[str],
                 company_type: str, 
                 experience_required: str, 
                 location: str, 
                 salary: str, 
                 current_step: int = 0, 
                 max_steps: int = 10, 
                 cumulative_reward: float = 0.0):
        self.candidate_skills = candidate_skills
        self.experience_level = experience_level
        self.job_description = job_description
        self.required_skills = required_skills
        self.company_type = company_type
        self.experience_required = experience_required
        self.location = location
        self.salary = salary
        self.current_step = current_step
        self.max_steps = max_steps
        self.cumulative_reward = cumulative_reward

    def to_dict(self) -> Dict:
        """Returns a serialized dictionary representation of the state for agents."""
        return {
            "candidate_skills": self.candidate_skills,
            "experience_level": self.experience_level,
            "job_description": self.job_description,
            "required_skills": self.required_skills,
            "company_type": self.company_type,
            "experience_required": self.experience_required,
            "location": self.location,
            "salary": self.salary,
            "current_step": self.current_step,
            "max_steps": self.max_steps,
            "cumulative_reward": self.cumulative_reward
        }
