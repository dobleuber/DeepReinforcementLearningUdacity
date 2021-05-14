import torch
import numpy as np
from collections import deque


def ddpg(env, agents, n_episodes=1000, max_t=300, print_every=100):
    scores_deque = deque(maxlen=print_every)
    scores = []

    brain_name = env.brain_names[0]

    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]
        for agent in agents:
            agent.reset()
        score = 0
        states = env_info.vector_observations

        for t in range(max_t):
            actions = [agent.act(states[i]) for i, agent in enumerate(agents)]
            env_info = env.step(actions)[brain_name]
            for i, agent in enumerate(agents):
                next_state = env_info.vector_observations[i]
                reward = env_info.rewards[i]
                done = env_info.local_done[i]
                agent.step(states[i], actions[i], reward, next_state, done)
            states = env_info.vector_observations
            score += np.mean(reward)

            if done:
                break

        scores_deque.append(score)
        scores.append(score)
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)), end="")
        torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
        torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
        if i_episode % print_every == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
