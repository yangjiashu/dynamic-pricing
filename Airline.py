# coding:utf-8
import numpy as np
import gym
from gym.spaces import Discrete, Box
import tensorflow as tf

class Airline(gym.Env):

    def __init__(self, state, prices, deadline):
        # 当创建实例时确定
        #assert isinstance(state['capacity'], int)
        #assert isinstance(prices, list)
        #assert isinstance(deadline, int)
        self.state = state  # 初始状态
        self.prices = prices  # 可行价格列表
        self.deadline = deadline  # 销售期限

        # 变量
        self.cur_capacity = self.state['capacity']  # 当前的剩余座位
        self.cur_price = self.prices[np.random.randint(0, len(self.prices))]  # 当前的价格
        self.time_remain = self.deadline  # 距起飞剩余时间

        self.action_space = Discrete(len(prices))
        self.observation_space = Discrete(state['capacity'])

    def reset(self):
        self.cur_capacity = self.state['capacity'] # 当前剩余座位重设为航班总座位
        #self.cur_price = self.prices[np.random.randint(0, len(self.prices))]
        self.time_remain = self.deadline # 剩余时间更改为销售期
        return self.cur_capacity

    def step(self, action):
        price = action # 设定价格
        if self.time_remain == 0: # terminal
            observation = -1
            reward = 0
            done = 1
            info = {}
        else: # not terminal
            # 通过demand function 来确定当前状态-价格下的随机订单数
            order = self.demand_func(self.cur_capacity, price, self.time_remain)
            # 确定下一个状态
            observation = (0 if self.cur_capacity < order else self.cur_capacity - order)
            # 确定即时回报
            reward = price * min(self.cur_capacity, order)
            #if done?
            done = 0
            info = {}

            self.time_remain -= 1
            self.cur_capacity = observation
        return observation, reward, done, info




    def render(self):
        pass

    def demand_func(self, capacity, a, time_remain):
        lam = 75 - 5 * time_remain * np.exp(-2 / time_remain * a / 100)
     #   if lam < 0 :
     #       lam = 0
     #   elif lam > 40:
     #       lam = 40
        return np.random.poisson(lam)