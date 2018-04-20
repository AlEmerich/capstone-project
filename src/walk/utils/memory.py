from collections import deque
import numpy as np

class Memory():
    """Implement the Experience Replay logic in order to
    train over samples of the memory at each pass and not just
    on the current state.
    """
    def __init__(self, length=1000):
        """Instantiate the queue
        """
        self.mem = deque(maxlen=legnth)

    def remember(self, state, action, reward, next_action, done):
        """Put a new SARS in the memory
        """
        self.mem.append([state, action, reward, next_action, done])

    def samples(self, batch_size):
        """Retrieve random batch_size of elements from the
        memory
        """
        indices = np.random.choice(np.arange(len(self.mem)),
                                   size=batch_size,
                                   replace=False)
        idx = [self.mem[i] for i in indices]
        return idx

    def __len__(self):
        return len(self.mem)
