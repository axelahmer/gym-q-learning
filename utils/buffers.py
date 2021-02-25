import torch


class ReplayBuffer(object):
    """
    A very simple memory buffer with some ideas taken from Berkley's assignment
    """

    def __init__(self, max_size, state_shape, state_dtype=torch.float32, device='cpu'):
        self.size = 0
        self.max_size = max_size
        self.next_idx = 0

        self.state = torch.empty([max_size] + list(state_shape), dtype=state_dtype, device=device)
        self.action = torch.empty([max_size], dtype=torch.int32, device=device)
        self.reward = torch.empty([max_size], dtype=torch.float32, device=device)
        self.next_state = torch.empty([max_size] + list(state_shape), dtype=state_dtype, device=device)
        self.done = torch.empty([max_size], dtype=torch.bool, device=device)

    def can_sample(self, batch_size):
        return batch_size <= self.size

    def sample(self, batch_size):
        assert self.can_sample(batch_size)

        i = torch.randperm(self.size)[:batch_size]
        return self.state[i], self.action[i], self.reward[i], self.next_state[i], self.done[i]

    def add(self, state, action, reward, next_state, done):
        """
        adds a transition to the buffer, overwriting if buffer full.
        """

        self.state[self.next_idx] = state
        self.action[self.next_idx] = action
        self.reward[self.next_idx] = reward
        self.next_state[self.next_idx] = next_state
        self.done[self.next_idx] = done

        self.next_idx = (self.next_idx + 1) % self.max_size

        if self.size < self.max_size:
            self.size += 1
