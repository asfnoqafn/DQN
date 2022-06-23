from random import seed
import gym
import torch
from actor import Agent
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
    
if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    agent = Agent(gamma= 0.99 ,epsilon= 1, state_size=[8],  batch_size= 64 ,action_size= 4, lr= 0.001, episode_end= 0.01)
    scores = []
    scores_window = deque(maxlen=100)
    n_tries = 2000


    for i in range(1,n_tries+1):
        score = 0
        done = False
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_new,reward ,done, info = env.step(action)
            score += reward
            agent.store_memory(observation,action,reward,observation_new, done)
            agent.learn()
            observation = observation_new
        scores_window.append(score) 
        scores.append(score)        
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i, np.mean(scores_window)), end="")
        if i % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i, np.mean(scores_window)))
        if np.mean(scores_window)>=200.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i-100, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            break
            
    


    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()



    