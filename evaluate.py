def evaluate(env, agent, episodes=200):
    total = 0
    for _ in range(episodes):
        state = env.reset()
        done = False
        while not done:
            action = agent.act(state)
            state, reward, done, _ = env.step(action)
            total += reward
    return total / episodes
