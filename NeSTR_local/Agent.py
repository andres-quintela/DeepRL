import torch.nn.functional as F
import gym
import torch as torch
from parameters import *
from replay_memory import *
import math as math
import random


class Agent(object):

	def __init__(self,env):

		self.env=env

		#initialize replay memory
		self.memory=replay_memory(MEMORY_SIZE)

		#initialize networks
		self.policy_net = DQN(action_size).to(device)
		self.target_net = DQN(action_size).to(device)
		target_net.load_state_dict(policy_net.state_dict())

		#initialize optimiser
		self.optimizer = optim.SGD(policy_net.parameters(),lr=LEARNING_RATE)

		#initialize action selector
		self.select_action = select_action

	def select_action(self,state):
	    sample = random.random()
	    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
	        math.exp(-1. * steps_done / EPS_DECAY)
	    if sample > eps_threshold:
	        with torch.no_grad():
	            return self.policy_net(state).max(1)[1].view(1, 1)
	    else:
	        return torch.tensor([[random.randrange(action_size)]], device=device, dtype=torch.long)




	def act_model(self,state,i_episode_reward):
		
		global i_episode_reward
		global steps_done
		action = select_action(state)
		next_state, reward, done, _ = self.env.step(action.item())
		reward = torch.tensor([reward], device=device)
        next_state=torch.tensor([next_state],dtype=torch.float,device=device)


        #accumulated reward for each episode
        i_episode_reward += reward.item()

        if done:
            next_state=None

        # Store the transition in memory
        self.memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

	def optimize_model(self):
		if len(memory) < BATCH_SIZE:
			return
		transitions = self.memory.sample(BATCH_SIZE)
			# Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
			# detailed explanation).
		batch = Transition(*zip(*transitions))

			# Compute a mask of non-final states and concatenate the batch elements
		non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
												  batch.next_state)), device=device, dtype=torch.uint8)
		non_final_next_states = torch.cat([s for s in batch.next_state
														if s is not None])
		state_batch = torch.cat(batch.state)
		action_batch = torch.cat(batch.action)
		reward_batch = torch.cat(batch.reward)

			# Compute Q(s_t, a) - the model computes Q(s_t), then we select the
			# columns of actions taken
		state_action_values = self.policy_net(state_batch).gather(1, action_batch)

			# Compute V(s_{t+1}) for all next states.
		next_state_values = torch.zeros(BATCH_SIZE, device=device)
		next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
			# Compute the expected Q values
		expected_state_action_values = (next_state_values * GAMMA) + reward_batch

			# Compute Huber loss
		self.loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

			# Optimize the model
		self.optimizer.zero_grad()
		self.loss.backward()
		for param in policy_net.parameters():
			param.grad.data.clamp_(-1, 1)
		self.optimizer.step()
