import numpy as np
from airline.Airline import Airline
from airline.RL_brain import QLearningTable
import pandas as pd

def update(env, RL):

    for episode in range(1000):
        RL.eligibility_trace *= 0
        observation = env.reset()
        total_reward = 0

        while True:
            env.render()

            action, exploit_state = RL.choose_action(observation)
            observation_, r, done, info = env.step(action)
            total_reward += r

            RL.learn(observation, action, r, observation_, exploit_state)

            observation = observation_

            if done:
                break
        if episode % 50 == 0:

            # total_reward = 0
            # observation = env.reset()
            # while True:
            #     action = RL.q_table.loc[observation,:]
            #     action = action.argmax()
            #     observation_, r, done, info = env.step(action)
            #     total_reward += (observation_ - observation) * action
            #     observation = observation_
            #     if done:
            #         break
            print(total_reward)
    RL.q_table.to_csv('q_table.csv',encoding='utf-8')
    total_reward = 0
    observation = env.reset()
    while True:
        action = RL.q_table.loc[observation,:]
        action = action.argmax()
        observation_, r, done, info = env.step(action)
        total_reward += (observation - observation_) * action
        observation = observation_
        if done:
            break
    print(total_reward)

# if __name__ == '__main__':
#     state = {'capacity': 50}
#     env = Airline(state, np.linspace(70, 120, 10), 10)
#     RL = QLearningTable(np.linspace(70, 120, 10), state)
#     update(env, RL)