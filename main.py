import gymnasium as gym
import random
import matplotlib.pyplot as plt
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


env=gym.make("CartPole-v1", render_mode = 'human')

# Buffer memory to store the past transitions
# From this memory the a batch of random transitions will be taken to train our agent
# Buffer Memory is stored in a deque data structure
class ReplayMemory(object) :
    def __init__(self,capacity) :
        self.memory = deque([], maxlen=capacity)

    def push(self, state, action, reward, next_state, done) :
        self.memory.append((state, action, reward, next_state, done))
    
    def sample(self,batch_size):
        states, actions, rewards, next_states, dones = zip(*random.sample(self.memory, batch_size))
        return np.vstack(states), actions, rewards, np.vstack(next_states), dones
    
    def current_size(self):
        return len(self.memory)
    
# Deep Q networkd with 2 hiden layers
class DQN(nn.Module) :
    def __init__(self,n_observation,n_action):
        super(DQN,self).__init__()
        self.layer1 = nn.Linear(n_observation, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_action)

    def forward(self,x) :   # forward pass of the network using the ReLU activation function
        x=torch.relu(self.layer1(x))
        x=torch.relu(self.layer2(x))
        return torch.relu(self.layer3(x))


#training parameters
batch_size = 128  #number of transition samples stored in the replay buffer for further study
discount_factor = 0.99
start_epsilon = 0.9
end_epsilon = 0.05
epsilon_decay = 0.995
learning_rate = 1e-4
period = 8   #time period of update of our target network

n_action = env.action_space.n
state, info = env.reset()
n_observation = len(state)



class DQNagent :
    def __init__(self, state_size, action_size) :
        self.state_size = state_size
        self.action_size = action_size
        self.memory = ReplayMemory(1000)
        self.q_network = DQN(n_observation, n_action)
        self.target_network = DQN(n_observation, n_action)
        self.optimize = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.epsilon = start_epsilon
        self.update_target()

    def update_target(self) :
        self.target_network.load_state_dict(self.q_network.state_dict())

    def action(self, state):
        if np.random.rand() <= self.epsilon :
            return random.choice(range(self.action_size))
        #conver the data types to tensors
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad(): #disable gradient calculations (because we are not training the agent at this stage)
            q_values = self.q_network(state)
        return torch.argmax(q_values, dim=1).item()
            
    def train(self, batch_size) :
        if batch_size > self.memory.current_size() :
            return 
        
        states, actions, rewards, next_state, done = self.memory.sample(batch_size)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_state = torch.FloatTensor(next_state)
        done = torch.FloatTensor(done)

        q_values = self.q_network(states)
        next_q_values = self.target_network(next_state)

        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_values.max(1)[0]
        expected_q_value = rewards + discount_factor * next_q_value * (1 - done)

        loss = nn.MSELoss()(q_value, expected_q_value)

        self.optimize.zero_grad()
        loss.backward()
        self.optimize.step()

    def store(self, state, action, reward, next_state, done) :
        self.memory.push(state, action, reward, next_state, done)

episodes = 100

# training loop
def train_agent():
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNagent(state_size, action_size)
    rewards = []

    for episode in range(episodes) :
        state, _ = env.reset()
        total_reward = 0
        while True :
            action = agent.action(state)
            next_state, reward, done, _, _ = env.step(action)
            agent.store(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            # for debuggging and plotting
            if done:
                rewards.append(total_reward)
                print(F"Episode : {episode}, Reward : {total_reward}")
                break
            agent.train(batch_size)
        if agent.epsilon > end_epsilon :
            agent.epsilon = agent.epsilon * epsilon_decay
        if episode % period == 0 :
            agent.update_target()
    return rewards


rewards = train_agent()


# For plotting decomment below

# plt.plot(rewards)
# plt.title("Cartpole Training with DQN")
# plt.xlabel("Episode")
# plt.ylabel("Reward")
# plt.savefig('results.png')





        







