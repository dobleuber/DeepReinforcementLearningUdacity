import torch
import numpy as np
import random
from collections import namedtuple, deque


class ReplayBuffer:
    """
    Fixed-size buffer to store experience tuples.
    """
    _singleton = None

    @staticmethod
    def get_replay_buffer():
        if ReplayBuffer._singleton is None:
            raise Exception('Should call set_replay_buffer first!')

        return ReplayBuffer._singleton

    @staticmethod
    def set_replay_buffer(action_size, buffer_size, batch_size, device):
        if ReplayBuffer._singleton is None:
            ReplayBuffer._singleton = ReplayBuffer(action_size, buffer_size, batch_size, device)

    def __init__(self, action_size, buffer_size, batch_size, device):
        """
        Initialize the replay buffer.
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple('Experience', field_names=['state', 'action', 'reward', 'next_state', 'done'])

        self.device = device

    def add(self, state, action, reward, next_state, done):
        """
        Add experience to the memory
        """
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """
        Gets a random batch of experience from memory
        """
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float()
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long()
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float()
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float()
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float()

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """
        Returns the current size of the memory
        """
        return len(self.memory)
