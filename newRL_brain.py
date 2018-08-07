import numpy as np
import pandas as pd

class Q_lam():
    def __init__(self, observations, actions, e_greedy = 0.9, reward_decay = 0.9, trace_decay = 0.9):
        self.observations = observations
        self.actions = actions
        self.epsilon = e_greedy
        self.gamma = reward_decay
        self.lam = trace_decay

        columns = actions
        index = list(range(1,observations['capacity'] + 1))

        self.q_table = pd.DataFrame(0, columns=columns, index=index)
        self.e_table = pd.DataFrame(0, columns=columns, index=index)

    def choose_action(self, observation):
        c = int(observation['capacity'])

        if np.random.uniform() > self.epsilon:
            state_action = self.q_table.loc[c, :]
            #state_action = state_action.reindex(np.random.permutation(state_action.index))
            action = state_action.argmax()
            is_exploit = 1
        else:
            action = np.random.choice(self.actions)
            is_exploit = 0

        return action, is_exploit

    def learn(self, s, a, r, s_, alpha, is_exploit, is_terminal):
        s = int(s['capacity'])
        s_ = int(s_['capacity'])

        q_predict = self.q_table.loc[s,a]
        if is_terminal:
            q_target = r
        else:
            q_target = r + self.gamma * self.q_table.loc[s_,:].max()
        error = q_target - q_predict

        self.e_table.loc[s,:] *= 0
        self.e_table.loc[s,a] = 1

        self.q_table += alpha * error * self.e_table
        # self.q_table += alpha * error
        self.e_table *= self.e_table * self.lam

