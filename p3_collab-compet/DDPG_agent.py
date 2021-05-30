import torch
import numpy as np
from collections import deque


def ddpg(env, agent, n_episodes=10000, print_every=100):
    scores_deque = deque(maxlen=100)
    scores_all = []
    brain_name = env.brain_names[0]

    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        agent.reset()
        scores = np.zeros(agent.num_agents)
        while True:
            actions = agent.act(states)
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            agent.step(states, actions, rewards, next_states, dones)
            states = next_states
            scores += rewards
            max_score = np.max(scores)

            if np.any(dones):
                break

        scores_deque.append(max_score)
        scores_all.append(max_score)
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)), end="")
        if i_episode % print_every == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))

        if np.mean(scores_deque) > 0.5:
            print('\nEnvironment solved in {:d} episodes! Mean score: {:.3f}'.format(
                i_episode, np.mean(scores_deque)))
            torch.save(agent.actor_local.state_dict(), 'trained_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'trained_critic.pth')
            break

    return scores_all
