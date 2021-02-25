import torch
import torch.nn as nn
from core.dqn import DQN


class SingleDQN(DQN):
    """
    My simplified implementation of Minh et al 2013 https://arxiv.org/pdf/1312.5602.pdf

    A DQN with only a single q network (no frozen target network).
    """

    def __init__(self, env, config):
        self.q_network = None
        self.optimizer = None
        super().__init__(env, config)

    def build(self):
        num_features = self.env.observation_space.shape[0]
        num_actions = self.env.action_space.n

        self.q_network = nn.Sequential(nn.Linear(num_features, self.config.hidden_units),
                                       nn.ReLU(),
                                       nn.Linear(self.config.hidden_units, num_actions)
                                       )

    def obs_to_state(self, obs):
        return torch.tensor(obs).float().to(self.device)

    def initialize(self):
        self.q_network.to(self.device)
        # TODO load weights
        self.optimizer = torch.optim.Adam(self.q_network.parameters())

    def get_best_action(self, state):
        with torch.no_grad():
            q_values = self.q_network(state)
        best_action = torch.argmax(q_values).item()
        return best_action, q_values

    def train_step(self, t, replay_buffer, lr_schedule):
        # sample minibatch from replay_buffer
        if replay_buffer.can_sample(self.config.batch_size):
            states, actions, rewards, next_states, done_mask = replay_buffer.sample(self.config.batch_size)

            # calculate loss
            num_actions = self.env.action_space.n
            gamma = self.config.gamma
            q_values = self.q_network(states)
            with torch.no_grad():
                target_q_values = self.q_network(next_states)
            # TODO try removing int64
            one_hot_actions = torch.nn.functional.one_hot(actions.to(torch.int64), num_actions).to(self.device)
            q = torch.sum(q_values * one_hot_actions, 1)
            q_target = torch.where(done_mask, rewards, rewards + gamma * torch.max(target_q_values, 1).values)
            mse_loss = torch.nn.functional.mse_loss(q, q_target.detach())

            # optimizer step
            self.optimizer.zero_grad()
            mse_loss.backward()
            for group in self.optimizer.param_groups:
                group['lr'] = lr_schedule.update(t)
            self.optimizer.step()

            return mse_loss.item()

        return 0.0