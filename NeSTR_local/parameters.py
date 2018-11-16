#Parameters#
import torch.optim as optim
import gym
import torch
from replay_memory	import *

env = gym.make('Acrobot-v1').unwrapped
action_size=env.action_space.n
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 128
GAMMA = 0.999

EPS_START = 0.99
EPS_END = 0.05
EPS_DECAY = 1000


TARGET_UPDATE = 10
LEARNING_RATE=0.1

MEMORY_SIZE=10000

num_episodes = 130
max_t=500  #maximum timesteps per episode
