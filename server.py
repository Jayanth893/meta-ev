from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
import sys

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from env.environment import JobApplyEnv

app = FastAPI(title="JobApplyEnv API")

# Initialize environment
# Use a default max_steps of 10 if not specified
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
    """Resets the environment and returns the initial state."""
    initial_state = env.reset()
    # Normalize state to match openenv.yaml if needed
    # (But usually just returning the dict is fine if it covers all fields)
    return initial_state

@app.post("/step", response_model=StepResponse)
def step(action: Action):
    """Executes a step in the environment."""
    try:
        # Convert Pydantic model to dict
        action_dict = action.model_dump()
        obs, reward, done, info = env.step(action_dict)
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
    """Returns the current state of the environment."""
    return env.state()

import uvicorn
import gradio as gr
from app import iface  # Import the Gradio interface

# Mount Gradio demo at /
app = gr.mount_gradio_app(app, iface, path="/")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
