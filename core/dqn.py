import torch
import numpy as np
from utils.schedules import LinearSchedule
from utils.buffers import ReplayBuffer


class DQN(object):
    """
    Abstract class for a DQN
    """

    def __init__(self, env, config):
        self.env = env
        self.config = config
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.build()

    def build(self):
        """
        build the models, called during construction.
        """
        raise NotImplementedError

    def initialize(self):
        """
        initializes dqn weights and parameters
        """
        raise NotImplementedError

    def obs_to_state(self, obs):
        """
        maps environment observations to MDP states, returns a torch tensor
        """
        raise NotImplementedError

    def get_best_action(self, state):
        """
        maps states -> best actions
        returns: action, q_values
        """
        raise NotImplementedError

    def get_egreedy_action(self, state, epsilon):
        """

        """
        action, q_values = self.get_best_action(state)
        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        return action, q_values

    def evaluate(self, num_episodes=1, render=False):
        """
        evaluate performance on environment with greedy policy. Does not update any learning parameters.
        returns average episode value.
        """
        scores = []

        for episode in range(num_episodes):

            # env = gym.wrappers.Monitor(env, "recording", force=True)
            obs = self.env.reset()
            state = self.obs_to_state(obs)
            score = 0

            while True:

                # interact with environment
                action, q_values = self.get_best_action(state)
                next_obs, reward, done, info = self.env.step(action)

                # render if needed
                self.env.render()

                # advance state
                obs = next_obs
                state = self.obs_to_state(obs)
                score += reward

                if done:
                    scores.append(score)
                    break

        return np.mean(scores)

    def train(self, exp_schedule: LinearSchedule, lr_schedule: LinearSchedule):
        """
        train model for n steps, following exp and lr schedule
        """
        replay_buffer = ReplayBuffer(self.config.buffer_size, self.env.observation_space.shape, device=self.device)
        num_steps = 0
        num_episodes = 0

        scores = []

        # main training loop
        while num_steps < self.config.train_nsteps:
            obs = self.env.reset()
            state = self.obs_to_state(obs)
            num_episodes += 1
            score = 0

            # environment/episode loop
            while True:
                num_steps += 1

                # get action
                epsilon = exp_schedule.update(num_steps)
                action, q_values = self.get_egreedy_action(state, epsilon)

                # interact with environment
                next_obs, reward, done, info = self.env.step(action)
                if self.config.render_train:
                    self.env.render()

                # save transition to memory
                next_state = self.obs_to_state(next_obs)
                replay_buffer.add(state, action, reward, next_state, done)

                # advance state
                state = next_state
                score += reward

                # train model if required
                loss = self.train_step(num_steps, replay_buffer, lr_schedule)

                # log step information
                # self.log(num_steps, q_values, loss)

                if done:
                    scores.append(score)
                    print(f'episode : {num_episodes} , score : {score}')
                    break
                if num_steps >= self.config.train_nsteps:
                    break

    def train_step(self, t, replay_buffer, lr_schedule):
        """
        How the agent is to train every step of environment interaction
        """
        raise NotImplementedError

    def run(self, lr_schedule: LinearSchedule, eps_schedule: LinearSchedule):
        """
        run the training loop
        """
        self.initialize()
        self.train(lr_schedule, eps_schedule)
        mean_score = self.evaluate(self.config.num_episodes_test, render=self.config.render_test)

        print(f'mean episode score : {mean_score}')
