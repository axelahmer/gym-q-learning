# Solving gym environments with q-learning in pyTorch

Slowly implmeneting different q-learning techniques to solve a bunch of simple gym environents. The main goals of this project are:
- increase my programming skills especially with pytorch
- learn some best practices with how to automate learning / logging / plotting etc.
- learn more about the history / development of DQNs
- learn about the trade-offs between models

## Dependencies:
- Python 3.8.5
- PyTorch 1.7.1
- Numpy 1.20.1
- Gym 0.18.0

## Implemented:

### Algorithms

- [x] Single network DQN (no frozen target network) - Minh et al 2013
- [ ] DQN (with frozen target network) - 2015
- [ ] Double DQN (avoids maximisation bias by using two q networks)
- [ ] Prioritised replay
- [ ] Duelling DQN

### Environments

- [x] CartPole-v1
- [x] Acrobot-v1
- [ ] MountainCar-v0
- [ ] Pendulum-v0

## Credits:
Code structure inspiration from Stanford's CS234 DQN Assignment: written by Guillaume Genthial and Shuhui Qu, updated by Haojun Li and Garrett Thomas.

Project idea flamed by Chris Yoon, who took a similar learning path. repo: https://github.com/cyoon1729/deep-Q-networks

My inspiration and knowledge of reinforcement learning has come from four main sources:
- Stanford's RL course
- David Silver's RL Course
- Reinforcement Learning: An Introduction (book) by Andrew Barto and Richard S. Sutton
- UC Berkley's CS188 Intro to AI course

## To-do:

- make graphs of:
    - loss per update
    - max/min/avg grad vs. grad step
    - eval score per score frequency
- Allow configs to specify deep q network architecture.
- Add all classical control encironments from gym
- save/load weights
- eventually figure out logging and tensorboard. (kinda the point of the project but SO boring.)

