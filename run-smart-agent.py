from unityagents import UnityEnvironment
from agent import Agent

import numpy as np
import time
import torch
import sys

MAX_T = 1000            # maximum number of timesteps per episode
N_EPISODES = 3          # Number of episodes

if __name__ == '__main__':
    env = UnityEnvironment(file_name="./Banana_Linux/Banana.x86_64")
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=False)[brain_name] # reset the environment

    action_size = brain.vector_action_space_size
    state = env_info.vector_observations[0]
    state_size = len(state)
    agent = Agent(state_size=state_size, action_size=action_size, seed=0)

    try:
        # load the weights from file
        agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))
    except Exception as e:
        print("Error: {}".format(e))
        print("Could not load checkpoint, you sure you trained the agent?")
        sys.exit(1)

    for i in range(N_EPISODES):
        env_info = env.reset(train_mode=False)[brain_name] # reset the environment
        state = env_info.vector_observations[0]            # get the current state
        score = 0                                          # initialize the score
        for t in range(MAX_T):
            action = agent.act(state)
            env_info = env.step(action)[brain_name]        # send the action to the environment
            state = env_info.vector_observations[0]        # get the next state
            reward = env_info.rewards[0]                   # get the reward
            done = env_info.local_done[0]                  # see if episode has
            score += reward
            time.sleep(0.05)
            if done:
                break
        print("Episode: {} - score: {}".format(i+1, score))
    env.close()
