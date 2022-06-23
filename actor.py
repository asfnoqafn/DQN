from qnetwork import *
import torch as T
import numpy as np

class Agent():
    def __init__(self, gamma, epsilon, lr, state_size, batch_size, action_size, max_memory_size = 100000, episode_end = 0.05 ,episode_decrement = 5e-4):
         self.gamma = gamma
         self.epsilon = epsilon
         self.lr = lr
         self.state_szie = state_size
         self.batch_size = batch_size
         self.action_space = [i for i in range(action_size)]
         self.max_memory_size =max_memory_size
         self.episode_min = episode_end
         self.episode_decrement = episode_decrement
         self.meomory_counter = 0
         self.qnetwork_local = QNetwork(self.lr, state_size = state_size, action_size = action_size, seed = 42)

         self.Q_eval = QNetwork(self.lr,state_size= state_size, action_size= action_size, seed= 0)

         self.state_memory = np.zeros((self.max_memory_size, *state_size),dtype= np.float32)
         self.new_state_memory = np.zeros((self.max_memory_size, *state_size),dtype= np.float32)
         self.action_memory = np.zeros(self.max_memory_size, dtype= np.int32)
         self.reward_memory = np.zeros(self.max_memory_size, dtype= np.float32)
         self.terminal_memory = np.zeros(self.max_memory_size, dtype= bool)


    # storing memory in arr of arrs
    def store_memory(self,state,action, reward, new_state, done):
        index = self.meomory_counter % self.max_memory_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.new_state_memory[index] = new_state
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.meomory_counter +=1

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = T.tensor(np.array(observation)).to(self.Q_eval.device)
            actions =   self.Q_eval.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
        return action

    def learn(self):
        # if amount sample less than batch size no point in learning
        if self.max_memory_size < self.batch_size:
            return

        self.Q_eval.optimizer.zero_grad()
        max_memory = min(self.max_memory_size, self.meomory_counter)

        batch =  np.random.choice(max_memory, self.batch_size)  
        batch_index = np.arange(self.batch_size, dtype = np.int32)
        state_batch = T.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = T.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
        reward_batch = T.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)
        action_batch = self.action_memory[batch]

        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        q_next = self.Q_eval.forward(new_state_batch)
        q_next[terminal_batch] = 0.0

        q_target = reward_batch + self.gamma * T.max(q_next, dim = 1)[0]

        loss = self.Q_eval.loss(q_target,q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()

        self.epsilon = self.epsilon - self.episode_decrement \
            if self.epsilon > self.episode_min else self.episode_min

        




