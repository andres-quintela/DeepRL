import matplotlib.pyplot as plt
from itertools import count
from network import *
from parameters import *
from Agent import *

episode_reward=[0]*num_episodes
i_episode_reward=0
i_episode=0
steps_done=0

if __name__ == '__main__':

	#initialize Agent
	Agent=Agent(env)
	for i_episode in range(num_episodes):
		# Initialize the environment and state
		state=env.reset()
		state=torch.tensor([state],dtype=torch.float,device=device)
		for t in count():
			Agent.act_model(state)
			Agent.optimize_model()
			steps_done += 1
			if t>max_t or done:
				episode_reward[i_episode]=i_episode_reward
				i_episode_reward=0
				break
			if i_episode % TARGET_UPDATE == 0:
				target_net.load_state_dict(policy_net.state_dict())

	plt.plot(episode_reward[:i_episode])
	plt.savefig('reward_episode.png')
