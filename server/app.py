from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from env.environment import JobApplyEnv

app = FastAPI(title="JobApplyEnv API")
env = JobApplyEnv(max_steps=10)

class Action(BaseModel):
    apply: bool
    resume_version: str

class StepResponse(BaseModel):
    observation: Dict[str, Any]
    reward: float
    done: bool
    info: Dict[str, Any]

@app.get("/health")
def health():
    return {"status": "ok", "env": "JobApplyEnv"}

@app.post("/reset")
def reset():
    return env.reset()

@app.post("/step", response_model=StepResponse)
def step(action: Action):
    try:
        obs, reward, done, info = env.step(action.model_dump())
        return {
            "observation": obs,
            "reward": float(reward),
            "done": bool(done),
            "info": info
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/state")
def state():
    return env.state()

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
