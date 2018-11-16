import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])

def get_screen(env, device):
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))  # transpose into torch order (CH
    # Convert to float, rescare, convert to torch tensor
    # (this doesn't require a copy)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
    return resize(screen).unsqueeze(0).to(device)

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    #saving a transition tuple
    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    #sample a random number according to batch size
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):

    def __init__(self, input_size,action_size):
        super(DQN, self).__init__()
        self.hidden1 = nn.Linear(input_size,16)
        self.hidden2 = nn.Linear(16,16)
        self.output = nn.Linear(16,action_size)

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        return self.output(x.view(x.size(0), -1))


class Agent :

    def __init__(self, device, task):
        self.device = device
        self.task = task
        self.env = gym.make(task).unwrapped

        self.steps_done = 0
        self.max_episodes = 150
        self.max_timesteps = 500
        self.epsilon_start = 0.99
        self.epsilon_end = 0.05
        self.epsilon_decay = 1000
        self.batch_size = 64
        self.gamma = 0.999
        self.target_update = 4
        self.learning_rate = 0.01
        self.TAU=0.001


        self.input_size=len(self.env.reset())
        self.action_size=self.env.action_space.n

        self.policy_network = DQN(self.input_size, self.action_size).to(device)
        self.target_network = DQN(self.input_size, self.action_size).to(device)
        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.target_network.eval()
        self.optimizer = optim.SGD(self.policy_network.parameters(),lr=self.learning_rate)
        self.memory = ReplayMemory(10000)

        self.episodes_rewards = []

    def get_epsilon_threshold(self):
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            math.exp(-1. * self.steps_done / self.epsilon_decay)

    def select_action(self, state):
        self.steps_done += 1
        if random.random() > self.get_epsilon_threshold():
            with torch.no_grad():
                return self.policy_network(state).max(1)[1].view(1, 1)
                
        else:
            return torch.tensor([[random.randrange(2)]], device=self.device, dtype=torch.long)
            
    #soft target update function
    def update_targets(self,target, original):
        """Weighted average update of the target network and original network
            Inputs: target actor(critic) and original actor(critic)"""
        
        for targetParam, orgParam in zip(target.parameters(), original.parameters()):
            targetParam.data.copy_((1 - self.TAU)*targetParam.data + self.TAU*orgParam.data)

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
        # detailed explanation).
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)), device=self.device, dtype=torch.uint8)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken
        state_action_values = self.policy_network(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        next_state_values[non_final_mask] = self.target_network(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_network.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        #soft update
        
        self.update_targets(self.target_network,self.policy_network)


    def train(self):
        self.episodes_rewards=[0]*self.max_episodes
        i_episode_reward=0

        for i_episode in range(self.max_episodes):
            print("episode : ", i_episode)

            # Initialize the environment and state
            state = self.env.reset()

            state = torch.tensor([state],dtype=torch.float,device=self.device)

            for t in count():
                # Select and perform an action
                action = self.select_action(state)
                next_state, reward, done, _ = self.env.step(action.item())
                #reward = torch.tensor([reward], device=self.device)

                #give intermediate reward
                if (next_state[0]< 0):
                    reward = torch.tensor([reward+1-next_state[0]],device=self.device)
                elif (next_state[0]>0):
                    reward = torch.tensor([reward+1-next_state[0]],device=self.device)
                else:
                    reward = torch.tensor([reward], device=self.device)

                next_state=torch.tensor([next_state],dtype=torch.float,device=self.device)


                #accumulated reward for each episode
                i_episode_reward += reward.item()

                if done:
                    next_state = None

                # Store the transition in memory
                self.memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the target network)
                self.optimize_model()
                if done or (t > self.max_timesteps):
                    #save episode reward
                    print(i_episode_reward)
                    self.episodes_rewards[i_episode]=i_episode_reward
                    i_episode_reward=0
                    break
        print('Complete')
        self.env.close()


    def display_training(self):
        plt.plot(self.episodes_rewards)
        plt.show()
