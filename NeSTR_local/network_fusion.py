import torch

from utils import Agent

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    game = 'Acrobot-v1'

    print("Creating an agent that plays ", game)
    agent = Agent(device, game)

    print("Train for ", agent.max_episodes, " episodes")
    agent.train()

    print("Display training")
    agent.display_training()

if __name__ == "__main__":
    main()
