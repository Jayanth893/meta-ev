from typing import Dict, Any, List, Set


class BaselineAgent:
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.bad_patterns = set()  # Memory to avoid repeated bad decisions

    def act(self, state: Dict[str, Any]) -> Dict[str, Any]:
        job_desc = state.get("job_description", "").lower()
        cand_skills = set(s.lower() for s in state.get("candidate_skills", []))
        req_skills = set(s.lower() for s in state.get("required_skills", []))
        cand_level_str = state.get("experience_level", "mid").lower()
        req_level_str = state.get("experience_required", "not specified").lower()
        company_type = state.get("company_type", "startup").lower()

        # 1. Skill Match Percentage
        skill_match_pct = 0.0
        if req_skills:
            intersection = cand_skills.intersection(req_skills)
            skill_match_pct = len(intersection) / len(req_skills)
        else:
            skill_match_pct = 0.5

        # 2. Experience Match
        exp_levels = {"junior": 1, "mid": 2, "senior": 3, "not specified": 2}
        cand_level = exp_levels.get(cand_level_str, 2)
        req_level = exp_levels.get(req_level_str, 2)
        
        exp_score = 0.0
        if cand_level < req_level:
            exp_score = -0.5
        elif cand_level == req_level:
            exp_score = 0.3
        else:
            exp_score = 0.1

        # 3. Company Preference
        company_score = 0.15 if company_type == "startup" else 0.0

        # Total Rule Score
        total_score = (skill_match_pct * 0.5) + exp_score + company_score

        # 4. Resume Version Selection
        frontend_keywords = {"react", "vue", "frontend"}
        backend_keywords = {"node", "python", "backend", "docker"}
        
        front_hits = sum(1 for kw in frontend_keywords if kw in job_desc)
        back_hits = sum(1 for kw in backend_keywords if kw in job_desc)

        if "fullstack" in job_desc:
            resume_version = "fullstack"
        elif front_hits > back_hits:
            resume_version = "frontend_focused"
        else:
            resume_version = "backend_focused"

        # Memory Check
        pattern = (req_level_str, company_type)
        if pattern in self.bad_patterns:
            total_score -= 0.4

        return {"apply": total_score >= self.threshold, "resume_version": resume_version}

    def feedback(self, state: Dict[str, Any], reward: float):
        if reward < 0:
            req_level_str = state.get("experience_required", "not specified").lower()
            company_type = state.get("company_type", "startup").lower()
            self.bad_patterns.add((req_level_str, company_type))
