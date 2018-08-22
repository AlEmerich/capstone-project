from collections import deque
import numpy as np


class Memory():
    """Implement the Experience Replay logic in order to
    train over samples of the memory at each pass and not just
    on the current state. Hold the queue of
    <state, action, reward, next_state, done>.
    """
    def __init__(self, length=1000):
        """Instantiate the queue
        """
        self.mem = deque(maxlen=length)

    def remember(self, state, action, reward, next_state, done):
        """Put a new SARS in the memory
        """
        self.mem.append([state, action, reward, next_state, done])

    def samples(self, batch_size):
        """Retrieve random batch_size of elements from the
        memory
        """
        idx = np.random.choice(len(self.mem),
                               size=batch_size,
                               replace=False)
        return self._stack_memory(np.array(self.mem)[idx])

    def _stack_memory(self, samples):
        """Separate the samples into numpy arrays of states, of
        actions, of rewards, of new_states and of dones in order
        to be in a good shape to feed neural network with it.
        """
        states = []
        actions = []
        rewards = []
        new_states = []
        dones = []
        for sample in samples:
            state, action, reward, new_state, done = sample
            states.append(np.array(state))
            actions.append(np.array(action))
            rewards.append(reward)
            new_states.append(np.array(new_state))
            dones.append(done)
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards).reshape(len(rewards), 1)
        new_states = np.array(new_states)
        dones = np.array(dones).reshape(len(dones), 1)

        return states, actions, rewards, new_states, dones

    def __len__(self):
        """return the length of the inner queue.
        """
        return len(self.mem)
