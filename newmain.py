from airline.newAirline import Airline
from airline.newRL_brain import Q_lam
import numpy as np

def update(env, RL, iter_times):
    for i in range(iter_times + 1):
        RL.e_table *= 0
        env.cur_observation = env.reset()
        total_reward = 0

        while True:
            env.render()
            alpha = 1 / (1 + i)
            observation = env.cur_observation
            action, is_exploit = RL.choose_action(env.cur_observation)
            observation_, r, is_terminal, info = env.step(action)
            total_reward += r
            RL.learn(observation, action, r, observation_, alpha, is_exploit, is_terminal)

            if is_terminal:
                break
        if i % 50 == 0:
            print(total_reward)

    RL.q_table.to_csv('q_table.csv', encoding='utf-8')


    # 测试
    total_reward = 0
    observation = env.reset()
    while True:
        action = RL.q_table.loc[int(observation['capacity']), :]
        action = action.argmax()
        observation_, r, done, info = env.step(action)
        total_reward += (int(observation['capacity']) - int(observation_['capacity'])) * action
        observation = observation_
        if done:
            break
    print('the optimal revenue is ' + str(total_reward))



if __name__ == '__main__':
    observation_space = {'capacity':100}
    action_space = np.linspace(70, 120, 5)
    deadline = 10
    iter_times = 1000

    env = Airline(observation_space, deadline)
    RL = Q_lam(observation_space, action_space, e_greedy=0.9, reward_decay=0.9, trace_decay=0.9)

    update(env, RL, iter_times)
