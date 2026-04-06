from env.environment import JobApplyEnv
from agents.baseline_agent import BaselineAgent
from tasks.easy import evaluate as easy_eval
from tasks.medium import evaluate as medium_eval
from tasks.hard import evaluate as hard_eval

env = JobApplyEnv()
agent = BaselineAgent()

print("Easy Score:", easy_eval(env, agent))
print("Medium Score:", medium_eval(env, agent))
print("Hard Score:", hard_eval(env, agent))
