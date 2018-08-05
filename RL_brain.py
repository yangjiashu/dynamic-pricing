import numpy as np
import pandas as pd

class QLearningTable:
    def __init__(self, actions, state, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9, trace_decay = 0.9):
        self.actions = actions  # a list
        self.lr = learning_rate  # 学习率
        self.gamma = reward_decay  # 奖励衰减
        self.epsilon = e_greedy  # 贪婪度
        self.lambda_ = trace_decay

        index = pd.Series(list(range(-1, state['capacity']+1)))
        #self.q_table = pd.DataFrame(0, columns=self.actions, index=index, dtype=np.float64)  # 初始 q_table
        self.q_table = pd.read_csv('q_table.csv')

        self.eligibility_trace = self.q_table.copy()

    def choose_action(self, observation):
        self.check_state_exist(observation)  # 检测本 state 是否在 q_table 中存在

        # 选择 action
        if np.random.uniform() < self.epsilon:  # 选择 Q value 最高的 action
            state_action = self.q_table.loc[observation, :]

            # 同一个 state, 可能会有多个相同的 Q action value, 所以我们乱序一下
            #np.random.shuffle(state_action)
            state_action = state_action.reindex(np.random.permutation(state_action.index))
            action = state_action.argmax()
            exploit_state = 1
        else:  # 随机选择 action
            action = np.random.choice(self.actions)
            exploit_state = 0

        return action, exploit_state

    def learn(self, s, a, r, s_, exploit_state):
        self.check_state_exist(s_)  # 检测 q_table 中是否存在 s_
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()  # 下个 state 不是 终止符
        else:
            q_target = r  # 下个 state 是终止符
        error = q_target - q_predict  # 更新对应的 state-action 值

        self.eligibility_trace.loc[s, :] *= 0
        self.eligibility_trace.loc[s, a] = 1

        self.q_table += self.lr * error * self.eligibility_trace

        if exploit_state == 1:
            self.eligibility_trace.loc[s, a] = self.lambda_ * self.eligibility_trace.loc[s, a]
        else:
            self.eligibility_trace.loc[s, a] = 0

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0] * len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )

            self.eligibility_trace = self.eligibility_trace.append(
                pd.Series(
                    [0] * len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )