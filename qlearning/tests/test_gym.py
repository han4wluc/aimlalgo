# import unittest
# from qlearning.Quantizer import Quantizer, _calc_size, _get_discrete_quantity
# from qlearning.QAgent import QAgent
# import gym
# import math
# import numpy as np

# ###################### Utility functions ####################################
# def get_explore_rate(t):
#     return max(MIN_EXPLORE_RATE, min(1.0, 1.0 - math.log10((t+1)/25)))

# def get_learning_rate(t):
#     return max(MIN_LEARNING_RATE, min(0.5, 1.0 - math.log10((t+1)/25)))
# #############################################################################

# class TestQuantizer(unittest.TestCase):

#   def test_constructor(self):

#     env = gym.make('CartPole-v0')

#     # print env.action_space.n

#     agent = QAgent(
#       actions=range(env.action_space.n),
#       # actions=[0,1],
#       alpha=0.1,
#       gamma=0.99,
#       explore_strategy='epsilon',
#       epsilon=0.1) # Create an agent


#     high = env.observation_space.high
#     low = env.observation_space.low
#     low[1] = -0.5
#     low[3] = -math.radians(50)
#     high[1] = 0.5
#     high[3] = math.radians(50)

#     quantizer = Quantizer(low=low,high=high,buckets=[1,1,6,3])


#     for n in range(50):
#       observation = env.reset()
#       for t in range(100):
#         action = env.action_space.sample()

#         # observation, reward, done, info = env.step(action)

#         quant_obs = quantizer.quantize(observation)

#         # print 'quant_obs', quant_obs
#         action, _ = agent.observe_and_act(
#           observation = quant_obs,
#           last_reward = 1
#         )
#         observation, reward, done, info = env.step(action)

#         # print 'reward', reward
#         # print 'observation', observation
#         # print 'discrete_obs', discrete_obs

#         if(done):
#           print 'Episode finished after {} timesteps'.format(t+1)
#           break

#     print agent.q_table


# if __name__ == '__main__':
#     unittest.main()



