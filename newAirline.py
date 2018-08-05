# -*- coding: UTF-8 -*-
import pandas as pd
import numpy as np
import gym

class Airline(gym.Env):
    def __init__(self, observations, deadline):

        # 原始状态
        self.observation = observations # dict
        self.deadline = deadline # int

        # 当前状态
        self.cur_observation = self.observation
        self.cur_time = 1

    def reset(self):

        #将当前状态和剩余时间重置
        self.cur_observation = self.observation
        self.cur_time = 1

        return self.cur_observation

    def step(self, action):

        order = self.demmand_func(self.cur_observation, self.cur_time, action)

        # terminal
        if (self.cur_observation['capacity'] - order) <= 0 or self.cur_time == self.deadline:
            observation_ = {'capacity': max(0, self.cur_observation['capacity'] - order)}
            self.cur_observation = observation_
            # observation_, reward, done, info
            return observation_, \
                   min(order, self.cur_observation['capacity']) * action, \
                   1, {}

        # not terminal
        else:
            observation_ = {'capacity': self.cur_observation['capacity'] - order}
            self.cur_observation = observation_
            self.cur_time += 1
            return observation_, \
                   order * action, \
                   0, {}

    def demmand_func(self, observation, t, a):
        return 75-5*t*np.exp(-2/t * a/100)

    def render(self, mode='human'):
        pass