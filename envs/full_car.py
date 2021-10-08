from gym import error, spaces
from gym import Env

try:
    import matlab.engine
    from matlab import double as double_m
except:
    print("No Matlab Engine.")

import math
import numpy as np
import random

import csv
import os
import copy
from ctypes import *
from sys import platform as _platform

from scipy.signal import welch

from utils.normalize import *
from utils.skyhook import skyhook
from utils.lqr import computeLQR

from utils.dual import calMA, dewel
from suspension_model.full_car_suspension import FullCarSuspension
from suspension_model.road_generator import RoadGenerator

import tensorflow as tf

class FullCar(Env):
    def __init__(self,  state_num=8, 
                        steps_per_episode=2000, 
                        args=None): 

        self.args = args

        self.state_num_wo_action = 31
        self.state_num = state_num

        if self.args.add_road_index == True:
            self.state_num += 1
            
        self.road_data = np.load('./misc/'+args.road_data+'.npz')
        self.observation_space = spaces.Box(low=-1, high=1, shape=(self.state_num*self.args.window_size,))

        self.action_size = self.args.action_size

        # Continuous Action Space
        self.action_space = spaces.Box(low=-1., high=1., shape=(self.action_size,))
        
        self.prefix = args.prefix

        self.episode_cnt = 0
        self.step_cnt = 0

        self.Cmax = 4000
        self.Cmin = 300

        self.state_SH = [0,0,0,0,0,0,0,0]

        self.steps_per_episode = int(steps_per_episode / (100 / self.args.sampling_freq))

        self.memory = [[0.0]*self.state_num] * self.args.window_size

        self.done = False
        self.info = {}

        # Settings for the impulse detector
        self.window_sm = 3
        self.window_out_sm = 20
        self.window_out_big = 200

        self.dewel_time = 200
        self.dewel_time_bump = 100

        self.in_acc_threshold_sm = 18
        self.in_acc_threshold_big = 80
        self.out_acc_threshold = 10

        self.model = FullCarSuspension()
        self.road_generator = RoadGenerator()


    def reset(self):
        self.bump_on_flag = False
        self.impact_bar = False
        if self.args.separate_training_target == "general_road":
            self.force_skyhook = self.bump_on_flag
        elif self.args.separate_training_target == "speed_bump":
            self.force_skyhook = not self.bump_on_flag

        self.acc_list_small_window = []
        self.acc_list_big_window = []

        self.bump_cnt = 0
        
        self.dewel_on_flag = False
        self.dewel_done = False

        self.step_cnt = 0
        self.dewel_count = 0

        self.temp_acc = []
        self.temp_phi = []
        self.temp_theta = []

        self.done = False

        self.memory = [[0.0]*self.state_num] * self.args.window_size

        return np.asarray([0.0] * self.args.window_size * self.state_num)            

    def step(self, action):
        while True:
                
            info = {}

            done = False
            
            obs_cand = self._compute(action, force_skyhook=self.force_skyhook)
            obs = [obs_cand[1], obs_cand[4], obs_cand[7], obs_cand[10], (obs_cand[1]-obs_cand[13]), (obs_cand[4]-obs_cand[16]), (obs_cand[7]-obs_cand[19]), (obs_cand[10]-obs_cand[22])]

            self.memorize(obs)
            self.memorizeBuffer(obs_cand)

            info['road_index'] = self.road_index[self.step_cnt][0]

            watch_target = (abs(info['raw_obs'][12])+abs(info['raw_obs'][15]))/2 # \ddot{z}_{COM}

            if self.bump_on_flag == False or self.impact_bar == True:
                ma = calMA(watch_target, self.acc_list_small_window, self.window_sm)

                if ma > self.in_acc_threshold_big:
                    # 작은 윈도우로 보고있다가 들어가기
                    # print("Bump detected", self.step_cnt, "step, Road idx:", info['road_index'], (abs(info['raw_obs'][0])+abs(info['raw_obs'][3]))/2, ma)
                    self.bump_cnt = 0
                    self.bump_on_flag = True
                    self.impact_bar = False
                    
            
            if self.bump_on_flag == True:

                ma = calMA(watch_target, self.acc_list_big_window, self.window_out_big)
                if ma <= self.out_acc_threshold:
                    # 큰 윈도우로 보다가 나가기
                    # print("Bump escaped", self.step_cnt, "step, Road idx:", info['road_index'], (abs(info['raw_obs'][0])+abs(info['raw_obs'][3]))/2)
                    self.bump_on_flag = False
                    self.impact_bar = False
                    self.dewel_on_flag = True
                    self.acc_list_big_window = []
                    self.acc_list_small_window = []

            if self.args.separate_training_target == "general_road":
                self.force_skyhook = self.bump_on_flag
            elif self.args.separate_training_target == "speed_bump":
                if self.impact_bar == True:
                    self.force_skyhook = True
                else:
                    self.force_skyhook = not self.bump_on_flag
            elif self.args.separate_training_target == "impact_bar":
                if self.impact_bar == True:
                    self.force_skyhook = not self.bump_on_flag
                else:
                    self.force_skyhook = True

            if self.bump_on_flag == True:
                self.bump_cnt += 1         

            reward = 0.
            for i in range(self.args.window_size):
                reward += self.getReward(self.memory_buffer[i],i)

            self.step_cnt += 1

            if self.step_cnt >= self.steps_per_episode:
                done = True
                self.step_cnt = 0
                self.episode_cnt += 1

                self.info = info
                self.done = done

            if self.force_skyhook == False or done == True or self.dewel_on_flag == True:
                break

        return np.asarray([item for sub in self.memory for item in sub]), reward, self.done, info

    def _compute(self, action, force_skyhook=False):
        def convertAction(action):
            action_scale = (action + 1)/2 # convert [-1,1] to [0,1]
            act0 = convertActionInnerFunc((self.state_SH[0] - self.state_SH[1]),action_scale[0])
            act1 = convertActionInnerFunc((self.state_SH[2] - self.state_SH[3]),action_scale[1])
            act2 = convertActionInnerFunc((self.state_SH[4] - self.state_SH[5]),action_scale[2])
            act3 = convertActionInnerFunc((self.state_SH[6] - self.state_SH[7]),action_scale[3])
            return [act0, act1, act2, act3]

        def convertActionInnerFunc(vel,scale):
            nominal_damping = 300
            if vel >= 0:
                a = nominal_damping * vel + 1000 * scale
            else:
                a = nominal_damping * vel - 1000 * scale
            return a

        action = convertAction(action)
        self.u_fl = action[0]
        self.u_fr = action[1]
        self.u_rl = action[2]
        self.u_rr = action[3]

        if self.dewel_on_flag == True:
            # Bump transient에서 빠져나옴
            if force_skyhook == True:
                # RL이 Bump Action임.
                [self.u_fl, self.u_fr, self.u_rl, self.u_rr], self.dewel_done, self.dewel_count = dewel([self.u_fl, self.u_fr, self.u_rl, self.u_rr], self.skyhook(), self.dewel_time, self.dewel_count)
            else:
                # RL이 General Road임
                [self.u_fl, self.u_fr, self.u_rl, self.u_rr], self.dewel_done, self.dewel_count = dewel(self.skyhook(), [self.u_fl, self.u_fr, self.u_rl, self.u_rr], self.dewel_time, self.dewel_count)
        else:
            if force_skyhook == True:
                [self.u_fl, self.u_fr, self.u_rl, self.u_rr] = self.skyhook()

        if self.dewel_done == True:
            self.dewel_on_flag = False

        self.road_FL = self.road_zl[self.step_cnt][0];self.road_FR = self.road_zr[self.step_cnt][0]
        obs = self.model.cal(u=[self.u_fl, self.u_fr, self.u_rl, self.u_rr], w=[self.road_FL, self.road_FR])

        self.state_SH = [obs["dz_fl"], obs["dz_tfl"], obs["dz_fr"], obs["dz_tfr"], obs["dz_rl"], obs["dz_trl"], obs["dz_rr"], obs["dz_trr"]]
        state = [obs["ddz_fl"], obs["dz_fl"], obs["z_fl"], obs["ddz_fr"], obs["dz_fr"], obs["z_fr"], obs["ddz_rl"], obs["dz_rl"], obs["z_rl"], obs["ddz_rr"], obs["dz_rr"], obs["z_rr"], obs["ddz_tfl"], obs["dz_tfl"], obs["z_tfl"], obs["ddz_tfr"], obs["dz_tfr"], obs["z_tfr"], obs["ddz_trl"], obs["dz_trl"], obs["z_trl"], obs["ddz_trr"], obs["dz_trr"], obs["z_trr"], obs["ddz"], obs["dz"], obs["z"], obs["dphi"], obs["phi"], obs["dtheta"], obs["theta"]]

        return state

    def getReward(self, obs, memory_index):

        if self.args.reward == 'acc_vel':
            return -((obs[0] - 0)**2 + (obs[1] - 0)**2 + obs[3]**2 + obs[4]**2 + obs[6]**2 + obs[7]**2 + obs[9]**2 + obs[10]**2 + obs[24]**2 + obs[25]**2)
        elif self.args.reward == 'com_jerk_pitch_acc':
            _last_com_acc = self.memory_buffer[memory_index+1][24]
            _current_com_acc = obs[24]

            _last_pitch = self.memory_buffer[memory_index+1][27]
            _current_pitch = obs[27]

            _timestep = 1 / self.args.sampling_freq

            _jerk = (_current_com_acc - _last_com_acc) * 10
            _pitch_acc = (_current_pitch - _last_pitch) * 10 
            return -((_jerk - 0)**2 + _pitch_acc**2)
        else:
            print("Wrong Reward")
            exit(0)
