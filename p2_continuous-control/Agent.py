import numpy as np
import random

import torch
import torch.nn.functional as F
import torch.optim as optim

from Actor import Actor
from Critic import Critic
from Noise import OUNoise
from ReplayBuffer import ReplayBuffer

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent:
    """
    Interacts with and learns from the environment.
    """
    _actor_local = None
    _actor_target = None
    _actor_optimizer = None

    @staticmethod
    def get_actor_local():
        if Agent._actor_local is None:
            raise Exception('Should call set_actors first!')
        return Agent._actor_local

    @staticmethod
    def get_actor_target():
        if Agent._actor_target is None:
            raise Exception('Should call set_actors first!')
        return Agent._actor_target

    @staticmethod
    def get_actor_optimizer():
        if Agent._actor_optimizer is None:
            raise Exception('Should call set_actors first!')
        return Agent._actor_optimizer

    @staticmethod
    def set_actors(state_size, action_size, random_seed):
        Agent._actor_local = Actor(state_size, action_size, random_seed).to(device)
        Agent._actor_target = Actor(state_size, action_size, random_seed).to(device)
        Agent._actor_optimizer = optim.Adam(Agent._actor_local.parameters(), lr=LR_ACTOR)

    _critic_local = None
    _critic_target = None
    _critic_optimizer = None

    @staticmethod
    def get_critic_local():
        if Agent._critic_local is None:
            raise Exception('Should call set_critics first!')
        return Agent._critic_local

    @staticmethod
    def get_critic_target():
        if Agent._critic_target is None:
            raise Exception('Should call set_critics first!')
        return Agent._critic_target

    @staticmethod
    def get_critic_optimizer():
        if Agent._critic_optimizer is None:
            raise Exception('Should call set_critics first!')
        return Agent._critic_optimizer

    @staticmethod
    def set_critics(state_size, action_size, random_seed):
        Agent._critic_local = Critic(state_size, action_size, random_seed).to(device)
        Agent._critic_target = Critic(state_size, action_size, random_seed).to(device)
        Agent._critic_optimizer = optim.Adam(Agent._critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

    def __init__(self, state_size, action_size, random_seed):
        """
        Initialize an Agent

        Params
        ======
            state_size (int): state dimension
            action_size (int): action dimension
            ramdon_seed (int): random seed
        """

        self.state_size = state_size
        self.action_size = action_size
        random.seed(random_seed)

        # Actor Network and its target network
        Agent.set_actors(state_size, action_size, random_seed)
        self.actor_local = Agent.get_actor_local()
        self.actor_target = Agent.get_actor_target()
        self.actor_optimizer = Agent.get_actor_optimizer()

        # Critic Network and its target network
        Agent.set_critics(state_size, action_size, random_seed)
        self.critic_local = Agent.get_critic_local()
        self.critic_target = Agent.get_critic_target()
        self.critic_optimizer = Agent.get_critic_optimizer()

        # Noise object
        self.noise = OUNoise(action_size, random_seed)

        # Replay Memory
        ReplayBuffer.set_replay_buffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
        self.memory = ReplayBuffer.get_replay_buffer()

    def step(self, state, action, reward, next_state, done):
        """
        Save experience in replay memory, and use random sample from buffer to learn.
        """

        # Save memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn from memory if enough samples exist
        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)

    def act(self, state, add_noise=True):
        """
        Returns actions for given state as per current policy.
        """
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()

        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()

        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """
        Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # update Critic
        # Get next predicted state, actions, and Q values
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)

        # Compute Q targets for current state
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Compute Critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update Actor
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()

        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update target networks
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)

    @staticmethod
    def soft_update(local_model, target_model, tau):
        """
        Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """

        for target_model, local_model in zip(target_model.parameters(), local_model.parameters()):
            target_model.data.copy_(tau * local_model + (1. - tau) * target_model.data)