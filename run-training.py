from unityagents import UnityEnvironment
from agent import Agent

import numpy as np
from collections import deque
import torch
import matplotlib.pyplot as plt

# DQN HYPERPARAMETERS
N_EPISODES = 2000       # maximum number of training episodes
MAX_T = 1000            # maximum number of timesteps per episode
EPS_START = 1.0         # starting value of epsilon, for epsilon-greedy action
                        # selection
EPS_END = 0.01          # minimum value of epsilon
EPS_DECAY = 0.995       # multiplicative factor (per episode) for decreasing
                        # epsilon
WINNING_REWARD = 13     # The problem is solved if the avarage score over 100
                        # episodes is at least WINNING_REWARD

def dqn(agent, env, brain_name):
    """Deep Q-Learning.

    Params
    ======
        - agent:
        - env:
        - brain_name
    """

    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = EPS_START                    # initialize epsilon
    for i_episode in range(1, N_EPISODES+1):
        env_info = env.reset(train_mode=True)[brain_name] # reset the environment
        state = env_info.vector_observations[0]            # get the current state

        score = 0
        for t in range(MAX_T):
            action = agent.act(state, eps)
            env_info = env.step(action)[brain_name]        # send the action to the environment
            next_state = env_info.vector_observations[0]   # get the next state
            reward = env_info.rewards[0]                   # get the reward
            done = env_info.local_done[0]                  # see if episode has finished

            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward

            if done:
                break

        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(EPS_END, EPS_DECAY*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window)>=WINNING_REWARD:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            break
    return scores

def plot_save(scores):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()
    fig.savefig('images/scores.png')   # save the figure to file

if __name__ == '__main__':

    # Initialize environment
    env = UnityEnvironment(file_name="./Banana_Linux/Banana.x86_64")

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # number of agents in the environment
    print('Number of agents:', len(env_info.agents))

    # number of actions
    action_size = brain.vector_action_space_size
    print('Number of actions:', action_size)

    # examine the state space
    state = env_info.vector_observations[0]
    print('States look like:', state)
    state_size = len(state)
    print('States have length:', state_size)

    agent = Agent(state_size=state_size, action_size=action_size, seed=0)

    scores = dqn(agent, env, brain_name)

    plot_save(scores)
