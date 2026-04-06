---
title: Meta-Ev
emoji: 🚀
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.4.0
python_version: "3.10"
app_file: app.py
pinned: false
---

# 🚀 JobApplyEnv

A realistic, AI-driven environment simulating real-world job application decisions. Train baseline or reinforcement learning agents to intelligently match candidate profiles with job postings and receive immediate rewards based on their decisions.

---

## 🎯 Overview

The **JobApplyEnv** provides a simulated workspace where an agent evaluates random job listings and determines if a candidate should apply. The candidate's skills and experience are evaluated against a real-world dataset of jobs (loaded via JSON). 

The goal is to maximize the application success rate by avoiding under-qualified rejections, penalizing applications to poorly-matched jobs, and confidently applying to perfectly-suited listings with the correct resume version.

## 🏗️ Architecture

```text
+-------------------+      State (Candidate + Job Data)      +-------------------+
|                   | -------------------------------------> |                   |
|   JobApplyEnv     |                                        |     AI Agent      |
|  (Environment)    | <------------------------------------- |  (Baseline / RL)  |
|                   |   Action (Apply?, Resume Version)      +-------------------+
+-------------------+                                                 |
        ^     |                                                       |
        |     | (Reward, Done State)                                  |
        |     +-------------------------------------------------------+
        |
+-------+-------+
| data/jobs.json| (External job records defining description, experience, skills)
+---------------+
```

## 🧩 State Space & Action Space

### Example State 
The agent receives an environment observation representing the current candidate and job parsing details:
```json
{
  "candidate_skills": ["JavaScript", "React", "Node.js", "MongoDB", "Docker"],
  "experience_level": "senior",
  "job_description": "Senior React architect: micro-frontends, module federation, Webpack",
  "company_type": "mnc",
  "experience_required": "senior",
  "location": "New York, NY",
  "salary": "$160k - $220k"
}
```

### Example Action
The agent must return an action object structuring the candidate's response:
```json
{
  "apply": true,
  "resume_version": "frontend_focused"
}
```

## 🧠 Detailed Reward Logic

JobApplyEnv calculates rewards by creating a `match_quality` score between the candidate and the job description. The final reward scales as follows based on the agent's application choice:

* **`+1.0` (Good Match + Apply):** The candidate possesses a strong skill match and their experience level aligns perfectly with the required experience.
* **`+0.5` (Partial Match + Apply):** The candidate has some overlapping skills but lacks specific seniority or minor domain keywords. 
* **`+0.3` (Smart Skip):** The job is a poor fit for the candidate, and the agent correctly chooses **not** to apply.
* **`-0.2` (Bad Apply):** The agent applies to a job where the candidate lacks the necessary core skills or falls significantly short in required experience.
* **`-0.5` (Missed Opportunity):** The candidate was a high-quality fit, but the agent mistakenly skipped the job listing.

## 📊 Sample Output

Running the evaluation script generates a comprehensive benchmark summary:
```text
==================================================
  Training RL Agent...
==================================================
Training completed for 20000 episodes.
Unique state-action pairs learned in Q-table: 34

==================================================
  Comparing Agents (10,000 evaluation episodes)
==================================================
Baseline Agent Mean Reward: 0.4671
RL Agent Mean Reward:       0.4669
--------------------------------------------------
Baseline outperformed RL Agent by 0.0002 average reward.
```

## 📸 Screenshots

> *(Replace with actual screenshots of your Gradio environment or plots)*

<!-- Placeholder for Screenshots -->
![Gradio Interface Placeholder UI](./docs/gradio_ui_placeholder.png)
> **Figure 1:** Interactive Gradio UI displaying job analytics and decision history.

## ⚙️ Setup & Execution

### 1. Execute Evaluator Tasks
To evaluate the baseline agent across task difficulties (`easy`, `medium`, `hard`):
```bash
python run.py
```

### 2. Compare Agents (Baseline vs Q-Learning)
To train the specialized reinforcement learning agent and benchmark it:
```bash
python compare_agents.py
```

### 3. Interactive Web Interface
Play alongside the agent or view live analytics via Gradio:
```bash
py app.py
```
*The UI now features a real-time **AI Recommendation** card powered by the trained RL agent.*

### 💾 Model Persistence
The RL agent now supports saving and loading its training state:
- **Save:** Automatically saved to `rl_model.json` after running `compare_agents.py`.
- **Load:** The Gradio app (`app.py`) automatically loads `rl_model.json` if it exists to provide intelligent recommendations.
