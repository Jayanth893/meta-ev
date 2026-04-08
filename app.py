import gradio as gr
import matplotlib.pyplot as plt
from env.environment import JobApplyEnv
from agents.rl_agent import QLearningAgent
import os

# Global environment instance to maintain state across interactions
env = JobApplyEnv()
agent = QLearningAgent()
if os.path.exists("rl_model.json"):
    agent.load_model("rl_model.json")

session_state = {
    "user_score": 0.0,
    "ai_score": 0.0,
    "episodes": 0,
    "user_reward_history": [],
    "ai_reward_history": [],
    "successful_apps": 0,
    "bad_decisions": 0,
    "ai_successful_apps": 0,
    "ai_bad_decisions": 0
}

def get_state_display():
    state = env.state()
    
    # 🎨 Transition Styles
    anim_css = """
<style>
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .card-container {
        animation: fadeIn 0.4s ease-out forwards;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .card-container:hover {
        transform: scale(1.015);
        box-shadow: 0 10px 20px rgba(0,0,0,0.1) !important;
        border-color: #2196f3 !important;
    }
    .skill-tag {
        transition: all 0.2s ease;
    }
    .skill-tag:hover {
        transform: translateY(-2px);
        filter: brightness(1.2);
        box-shadow: 0 4px 8px rgba(33, 150, 243, 0.3);
    }
</style>
"""
    
    # Format Job Description Card
    job_md = f"""
{anim_css}
<div class="card-container" style="padding: 24px; border-radius: 16px; border: 1px solid #90caf9; border-left: 8px solid #0d47a1; background-color: #bbdefb; height: 100%; box-shadow: 2px 2px 10px rgba(0,0,0,0.05); cursor: default;">
    <h3 style="color: #0d47a1; margin-top: 0; letter-spacing: 1px; font-weight: 800;">🏢 {state['company_type'].upper()} COMPANY</h3>
    <div style="margin: 12px 0;">
        <div style="color: #1a237e; margin-bottom: 4px;"><b>📍 Location:</b> {state.get('location', 'Remote')}</div>
        <div style="color: #004d40; margin-bottom: 4px;"><b>💼 Salary:</b> {state.get('salary', 'Competitive')}</div>
        <div style="color: #01579b;"><b>🎓 Exp Required:</b> {state.get('experience_required', 'Not Specified')}</div>
    </div>
    <hr style="border: 0; border-top: 1px solid #90caf9; margin: 18px 0;">
    <b style="color: #0d47a1; font-size: 14px; text-transform: uppercase;">Role Details:</b>
    <p style="font-size: 16px; color: #1a237e; line-height: 1.6; margin-top: 8px;">{state['job_description']}</p>
</div>
"""
    
    # Format Candidate Card
    cand_md = f"""
{anim_css}
<div class="card-container" style="padding: 24px; border-radius: 16px; border: 1px solid #bbdefb; border-left: 8px solid #2196f3; background-color: #e3f2fd; height: 100%; box-shadow: 2px 2px 10px rgba(0,0,0,0.05); cursor: default;">
    <h3 style="color: #0d47a1; margin-top: 0; font-weight: 800;">👤 CANDIDATE PROFILE</h3>
    <div style="margin: 15px 0;">
        <div style="color: #1a237e; margin-bottom: 4px;"><b>🎓 Experience:</b> {state['experience_level'].title()}</div>
        <div style="color: #1565c0; margin-bottom: 4px;"><b>📍 Prefers:</b> {env.candidate['preferred_location']}</div>
        <div style="color: #00695c; margin-bottom: 4px;"><b>💰 Target:</b> ${env.candidate['salary_expectation']:,}</div>
    </div>
    <b style="color: #1565c0; font-size: 14px; text-transform: uppercase;">Top Skills:</b><br/>
    <div style="margin-top: 12px;">
        {' '.join([f'<code class="skill-tag" style="background: #1565c0; color: white; padding: 6px 12px; border-radius: 6px; margin-right: 6px; display: inline-block; margin-bottom: 8px; font-weight: 500; font-size: 13px; cursor: pointer;">{s}</code>' for s in state['candidate_skills']])}
    </div>
</div>
"""
    # Get Match stats for display
    cand_skills = set(s.lower() for s in state['candidate_skills'])
    req_skills = set(s.lower() for s in env.current_job.get("required_skills", []))
    match_pct = 0
    if req_skills:
        intersection = cand_skills.intersection(req_skills)
        match_pct = (len(intersection) / len(req_skills)) * 100

    # Get Agent Recommendation
    agent_action = agent.act(state)
    rec_color = "#28a745" if agent_action["apply"] else "#6c757d"
    rec_icon = "✅ Apply" if agent_action["apply"] else "⏭️ Skip"
    
    agent_md = f"""
<div style="margin-top: 20px; padding: 15px; border-radius: 12px; background: #f8f9fa; border: 1px dashed #ced4da;">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
        <div style="font-size: 14px; color: #6c757d; font-weight: bold; text-transform: uppercase; letter-spacing: 0.5px;">🎯 Skill Match</div>
        <div style="font-size: 18px; font-weight: 800; color: {'#28a745' if match_pct > 70 else '#ffa000' if match_pct > 40 else '#dc3545'};">{match_pct:.0f}%</div>
    </div>
    <div style="font-size: 14px; color: #6c757d; margin-bottom: 8px; font-weight: bold; text-transform: uppercase; letter-spacing: 0.5px;">🤖 AI Recommendation</div>
    <div style="display: flex; align-items: center; gap: 10px;">
        <span style="font-size: 20px; font-weight: 800; color: {rec_color};">{rec_icon}</span>
        <span style="color: #6c757d; font-size: 14px;">(using {agent_action['resume_version']} resume)</span>
    </div>
</div>
"""
    return job_md, cand_md, agent_md

def generate_plot():
    fig, ax = plt.subplots(figsize=(8, 3))
    user_y = session_state["user_reward_history"]
    ai_y = session_state["ai_reward_history"]
    x_data = list(range(1, len(user_y) + 1))
    
    if user_y:
        ax.plot(x_data, user_y, marker='o', linestyle='-', color='#0d6efd', linewidth=2, label="You")
    if ai_y:
        ax.plot(x_data, ai_y, marker='x', linestyle='--', color='#6c757d', linewidth=1.5, alpha=0.8, label="AI Agent")
        
    ax.set_title("Performance Timeline (Recent Decisions)")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Reward")
    ax.set_ylim(-1.1, 1.6)
    ax.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend(loc="upper left", frameon=False, fontsize='small')
    
    plt.tight_layout()
    return fig

def take_action(action_type, resume_ver, mode):
    # Current state
    state = env.state()
    agent_action = agent.act(state)
    
    # User action
    user_action = {
        "apply": action_type == "apply",
        "resume_version": resume_ver
    }
    
    # If in AI Agent mode, the "User" choice is forced to the Agent choice
    effective_user_action = agent_action if mode == "AI Agent" else user_action
    
    # Calculate rewards for both
    # Note: we call calculate_reward directly to avoid advancing the environment twice
    user_reward = env.calculate_reward(effective_user_action)
    ai_reward = env.calculate_reward(agent_action)
    
    # Step environment with the effective action to advance
    _, _, done, _ = env.step(effective_user_action)
    
    # Update Analytics
    session_state["user_score"] += user_reward
    session_state["ai_score"] += ai_reward
    session_state["user_reward_history"].append(user_reward)
    session_state["ai_reward_history"].append(ai_reward)
    
    # Increment UI episode count only when the whole episode is finished
    if done:
        session_state["episodes"] += 1
    
    if effective_user_action["apply"] and user_reward > 0:
        session_state["successful_apps"] += 1
    if user_reward < 0:
        session_state["bad_decisions"] += 1
        
    if agent_action["apply"] and ai_reward > 0:
        session_state["ai_successful_apps"] += 1
    if ai_reward < 0:
        session_state["ai_bad_decisions"] += 1
    
    # Provide feedback to the agent
    agent.feedback(state, ai_reward)
        
    if len(session_state["user_reward_history"]) > 20:
        session_state["user_reward_history"] = session_state["user_reward_history"][-20:]
        session_state["ai_reward_history"] = session_state["ai_reward_history"][-20:]
    
    # Formatting
    color = "#28a745" if user_reward > 0 else "#dc3545" if user_reward < 0 else "#6c757d"
    ai_color = "#28a745" if ai_reward > 0 else "#dc3545" if ai_reward < 0 else "#6c757d"
    
    reward_md = f"""
<div style='display: flex; gap: 20px; animation: fadeIn 0.5s ease-out;'>
    <div style='flex: 1; border-left: 6px solid {color}; padding: 15px; background: rgba(0,0,0,0.02); border-radius: 0 10px 10px 0;'>
        <div style='font-size: 14px; color: #666; font-weight: bold;'>YOUR REWARD</div>
        <h1 style='color:{color}; margin: 0; font-size: 36px;'>{user_reward:+.1f}</h1>
        <div style='font-size: 12px; color: {color}77;'>{"Great Choice!" if user_reward > 0.5 else "Keep trying..." if user_reward < 0 else "Neutral"}</div>
    </div>
    <div style='flex: 1; border-left: 6px solid {ai_color}; padding: 15px; background: rgba(0,0,0,0.02); border-radius: 0 10px 10px 0; opacity: 0.8;'>
        <div style='font-size: 14px; color: #666; font-weight: bold;'>AI REWARD</div>
        <h1 style='color:{ai_color}; margin: 0; font-size: 36px;'>{ai_reward:+.1f}</h1>
        <div style='font-size: 12px; color: {ai_color}77;'>({agent_action['resume_version']})</div>
    </div>
</div>
"""
    
    score_md = f"""
<div style="background: white; padding: 20px; border-radius: 16px; border: 1px solid #e0e0e0; box-shadow: 0 4px 12px rgba(0,0,0,0.05);">
    <div style="display: flex; justify-content: space-between; margin-bottom: 20px;">
        <div style="text-align: center; flex: 1;">
            <div style="font-size: 24px;">👤</div>
            <div style="font-weight: 800; color: #1a73e8; font-size: 28px;">{session_state['user_score']:.1f}</div>
            <div style="font-size: 10px; color: #777; text-transform: uppercase;">Total Score</div>
        </div>
        <div style="text-align: center; border-left: 1px solid #eee; border-right: 1px solid #eee; flex: 1; padding: 0 10px;">
             <div style="font-size: 18px; color: #666; margin-top: 10px;">Episode <b>{session_state['episodes']}</b></div>
             <div style="font-size: 12px; color: #999;">Step {env.current_step}/{env.max_steps}</div>
        </div>
        <div style="text-align: center; flex: 1;">
            <div style="font-size: 24px;">🤖</div>
            <div style="font-weight: 800; color: #5f6368; font-size: 28px;">{session_state['ai_score']:.1f}</div>
            <div style="font-size: 10px; color: #777; text-transform: uppercase;">Total Score</div>
        </div>
    </div>
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; font-size: 12px;">
        <div style="background: #f1f3f4; padding: 10px; border-radius: 8px;">
            <b>Success:</b> {session_state['successful_apps']}<br/>
            <b>Mistakes:</b> {session_state['bad_decisions']}
        </div>
        <div style="background: #f1f3f4; padding: 10px; border-radius: 8px;">
            <b>Success:</b> {session_state['ai_successful_apps']}<br/>
            <b>Mistakes:</b> {session_state['ai_bad_decisions']}
        </div>
    </div>
</div>
"""
    
    # Prepare next state
    if done:
        env.reset()
        
    job_md, cand_md, agent_md = get_state_display()
    return cand_md, job_md, agent_md, reward_md, score_md, generate_plot()

def reset_game():
    session_state["user_score"] = 0.0
    session_state["ai_score"] = 0.0
    session_state["episodes"] = 0
    session_state["user_reward_history"] = []
    session_state["ai_reward_history"] = []
    session_state["successful_apps"] = 0
    session_state["bad_decisions"] = 0
    session_state["ai_successful_apps"] = 0
    session_state["ai_bad_decisions"] = 0
    
    env.reset()
    job_md, cand_md, agent_md = get_state_display()
    
    score_md = f"""
<div style="background: white; padding: 20px; border-radius: 16px; border: 1px solid #e0e0e0; box-shadow: 0 4px 12px rgba(0,0,0,0.05); text-align: center;">
    <h3 style="color: #666;">Ready to start!</h3>
    <p style="color: #999; font-size: 14px;">Select your action below to compete against the RL agent.</p>
</div>
"""
    waiting_md = "<h2 style='margin: 0; color: #6c757d; font-style: italic;'>Awaiting your first decision...</h2>"
    
    return cand_md, job_md, agent_md, waiting_md, score_md, generate_plot()

# Build Gradio UI
custom_css = """
footer {visibility: hidden}
.gradio-container {
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
}
.apply-btn {
    background: linear-gradient(135deg, #28a745 0%, #218838 100%) !important;
    border: none !important;
    box-shadow: 0 4px 15px rgba(40, 167, 69, 0.3) !important;
}
.skip-btn {
    background: linear-gradient(135deg, #6c757d 0%, #5a6268 100%) !important;
    border: none !important;
}
"""

with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", font=[gr.themes.GoogleFont("Inter"), "sans-serif"]), css=custom_css) as iface:
    gr.HTML("""
<div style="text-align: center; padding: 40px 0; background: linear-gradient(135deg, #0d47a1 0%, #1976d2 100%); color: white; border-radius: 24px; margin-bottom: 30px; box-shadow: 0 10px 30px rgba(13, 71, 161, 0.2);">
    <h1 style="font-size: 42px; margin: 0; font-weight: 900; letter-spacing: -1px;">🚀 Meta-Ev: Job Search RL</h1>
    <p style="font-size: 18px; opacity: 0.9; margin-top: 10px; font-weight: 400;">Can you outperform a trained Reinforcement Learning agent in the job market?</p>
</div>
""")

    with gr.Row():
        with gr.Column(scale=3):
            with gr.Row():
                with gr.Column():
                    cand_box = gr.HTML()
                with gr.Column():
                    job_box = gr.HTML()
            
            with gr.Row(variant="compact"):
                 with gr.Column(scale=2):
                    resume_ver = gr.Dropdown(
                        choices=["general", "frontend_focused", "backend_focused", "fullstack"],
                        value="general",
                        label="📄 Choose Resume Version",
                        info="Matching the resume to the job helps improve match quality."
                    )
                 with gr.Column(scale=3):
                    with gr.Row():
                        apply_btn = gr.Button("✅ Apply to Job", variant="primary", elem_classes=["apply-btn"], size="lg")
                        skip_btn = gr.Button("⏭️ Skip Listing", variant="secondary", elem_classes=["skip-btn"], size="lg")
            
        with gr.Column(scale=2):
            score_display = gr.HTML()
            reward_display = gr.HTML()
            agent_md_box = gr.HTML()
            
            mode_toggle = gr.Radio(
                choices=["Manual Mode", "AI Agent"], 
                value="Manual Mode", 
                label="🎮 Interaction Mode",
                info="In AI mode, your decisions are replaced by the RL agent's optimal choices."
            )

    with gr.Row():
        plot_box = gr.Plot(label="Comparison History (Recent Rewards)")
        
    gr.Markdown("---")
    with gr.Row():
        gr.Markdown("### 🛠️ Lab Controls")
        reset_btn = gr.Button("🔄 Reset Simulation & Scores", size="sm")

    # Wire up logic
    apply_btn.click(
        fn=lambda rv, mode: take_action("apply", rv, mode), 
        inputs=[resume_ver, mode_toggle], 
        outputs=[cand_box, job_box, agent_md_box, reward_display, score_display, plot_box]
    )
    
    skip_btn.click(
        fn=lambda rv, mode: take_action("skip", rv, mode), 
        inputs=[resume_ver, mode_toggle], 
        outputs=[cand_box, job_box, agent_md_box, reward_display, score_display, plot_box]
    )
    
    reset_btn.click(
        fn=reset_game,
        inputs=[],
        outputs=[cand_box, job_box, agent_md_box, reward_display, score_display, plot_box]
    )
    
    # Load initial state
    iface.load(
        fn=reset_game,
        inputs=[],
        outputs=[cand_box, job_box, agent_md_box, reward_display, score_display, plot_box]
    )

def main():
    """Main entry point for the Gradio interface."""
    iface.launch()

if __name__ == "__main__":
    main()
