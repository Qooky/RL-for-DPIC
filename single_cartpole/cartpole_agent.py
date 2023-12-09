import gym
import numpy as np
import matplotlib.pyplot as pltq
import time

class CartPoleEnv(gym.Env):

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 50}

    def __init__(self, render_mode="rgb_array", n_observations=1000, q_table=np.array(None)):
      self.env = gym.make("CartPole-v1", render_mode=render_mode)
      self.action_space = self.env.action_space
      self.observation_space = self.env.observation_space
      self.n_observation_space = 4
      self.n_buckets = 16
      self.bucket_size = self.env.observation_space.high / (self.n_buckets/2)
      self.q_table =  np.zeros((self.env.action_space.n, np.power(self.n_buckets, self.n_observation_space))) if not q_table.all() else q_table
      self.epsilon = 1
      self.n_episodes = n_observations
      self.epsilon_delta = self.epsilon / self.n_episodes
      self.metric = []
      self.alpha = 0.5
      self.gamma =  0.99
      return 
    
    def get_index(self, observation):
      observation_bucketed = self.bucket_observation(observation)
      i = len(observation_bucketed) - 1
      idx = 0
      for el in observation_bucketed:
        idx = idx + (self.n_buckets**i * el)
        i -= 1
      return int(idx)
    
    def bucket_observation(self, observation):
      return(np.floor(observation/self.bucket_size))

    def update_q_table(self, observation, observation_prime, reward, action=1):
      Q = self.q_table[action][self.get_index(observation)]
      Q = (1-self.alpha)*Q + self.alpha*(reward + self.gamma*max(self.q_table[a][self.get_index(observation_prime)] for a in range(2))) 
      self.q_table[action][self.get_index(observation)] = Q

    def train(self):
        observation, _ = self.env.reset()
        j=0
        for i in range(self.n_episodes):
          action = self.policy(observation)
          observation_prime, reward, terminated, truncated, info = self.env.step(action)
          self.update_q_table(observation, observation_prime, reward, action)
          j +=1
          if terminated or truncated:
            self.metric.append(j)
            j=0
            observation, info = self.env.reset()
            # print(info)
        self.env.close()
      
    def test(self):
        self.env.unwrapped.render_mode = "human"
        observation, _ = self.env.reset()
        for i in range(1000):
            index = self.get_index(observation)
            if self.q_table[0][index] > self.q_table[1][index]:
              action = 0
            else:
              action = 1 
            observation, reward, terminated, truncated, info = self.env.step(action)
            if terminated or truncated:
              time.sleep(0.1)
              observation, _ = self.env.reset()
        self.env.reset()
        self.env.close()
            
    def policy(self, observation):
      self.epsilon = self.epsilon - self.epsilon_delta   #at first, low probability to read from q-table, ie high prob take random action
      take_random_action = self.epsilon < np.random.random()
      if take_random_action:
         return self.action_space.sample()
      else:
         index = self.get_index(observation)
         if self.q_table[0][index] > self.q_table[1][index]:
           return 0
         else:
           return 1
         
def main():
    agent =  CartPoleEnv(render_mode="rgb_array", n_observations=100000)
    agent.train()
    agent.test()
    plt.plot(agent.metric)

if __name__ == "__main__":
    main()
       

