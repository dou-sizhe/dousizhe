import numpy as np
import collections


class ReplayBuffer:
    def __init__(self, args):
        self.size = args.buffer_size
        self.buffer = collections.deque(maxlen=self.size)

    def store_episode(self, episode):
        self.buffer.append(episode)

    def sample(self, batch_size):
        temp_buffer = []
        idx = np.random.randint(0, self.len(), batch_size)
        for i in idx:
            temp_buffer.append(self.buffer[i])
        return temp_buffer

    def len(self):
        return len(self.buffer)
