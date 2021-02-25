import gym
from core.single_dqn import SingleDQN
from utils.schedules import LinearSchedule

from configs.cartpole import config

if __name__ == '__main__':

    # make env
    env = gym.make(config.env_name)

    # exploration strategy
    exp_schedule = LinearSchedule(config.eps_begin,
                                  config.eps_end, config.eps_nsteps)

    # learning rate schedule
    lr_schedule = LinearSchedule(config.lr_begin, config.lr_end,
                                 config.lr_nsteps)

    # train model
    model = SingleDQN(env, config)
    model.run(exp_schedule, lr_schedule)
