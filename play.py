import sys

from stable_baselines import SAC

from envs.full_car_env import FullCar
from utils.arg_parser import common_arg_parser, parseLayersFromArgs
from utils.dual import calMA, dewel

import os
from glob import glob

import numpy as np

import csv
import re

import json

from PyInquirer import prompt, print_json
from argparse import Namespace

def main(args):
    env = FullCar(args=args)

    layers = parseLayersFromArgs(args=args) # default [32, 32]
    policy_kwargs = dict(layers=layers)

    model = SAC.load(args.model_path, env=env)
    bump_model = SAC.load(args.second_model_path, env=env)

    test_runs = 50
    
    ma_list = []

    window_sm = 3
    window_out_sm = 20
    window_out_big = 200

    dewel_time = 200
    dewel_time_bump = 100

    in_acc_threshold_sm = 18
    in_acc_threshold_big = 80
    out_acc_threshold = 10

    for i in range(test_runs):
        obs = env.reset()

        print(i+1,"/",test_runs)

        ma = 0
        acc_list_small_window = []
        acc_list_big_window = []
        acc_list_watch = []

        bump_cnt = 0
        bump_on_flag = False
        dewel_on_flag = False
        dewel_done = False
        dewel_bump_done = False

        step_cnt = 0
        dewel_count = 0
        dewel_count_bump = 0
        while True:
            action, _states = model.predict(obs)
            bump_action, _states_ = bump_model.predict(obs)

            if dewel_on_flag == True:
                action, dewel_done, dewel_count = dewel(bump_action, action, dewel_time, dewel_count)

            if dewel_done == True:
                dewel_on_flag = False

            if bump_on_flag == True:
                obs, reward, done, info = env.step(bump_action)
            else:
                obs, reward, done, info = env.step(action)

            try:
                watch_target = (abs(info['raw_obs'][12])+abs(info['raw_obs'][15]))/2 # \ddot{z}_{COM}
            except:
                print("Info 없음")
                watch_target = 0.

            
            if bump_on_flag == False:
                ma = calMA(watch_target, acc_list_small_window, window_sm)
                if ma > in_acc_threshold_big:
                    # 작은 윈도우로 보고있다가 들어가기
                    print("Bump detected", step_cnt, "step, Road idx:", info['road_index'], (abs(info['raw_obs'][0])+abs(info['raw_obs'][3]))/2, ma)
                    bump_cnt = 0
                    bump_on_flag = True
                    acc_list_small_window = []
            
            if bump_on_flag == True:
                ma = calMA(watch_target, acc_list_big_window, window_out_big)
                if ma <= out_acc_threshold:
                    # 큰 윈도우로 보다가 나가기
                    print("Bump escaped", step_cnt, "step, Road idx:", info['road_index'], (abs(info['raw_obs'][0])+abs(info['raw_obs'][3]))/2)
                    bump_on_flag = False
                    dewel_on_flag = True
                    acc_list_big_window = []

            ma_list.append(ma)

            if bump_on_flag == True:
                bump_cnt += 1                
            
            step_cnt += 1

            if done:
                break


if __name__ == '__main__':
    arg_parser = common_arg_parser()
    args, unknown_args = arg_parser.parse_known_args(sys.argv)

    # log_files = sorted(glob(logDir()+"/*"))
    log_files = sorted(glob('final_models/*'))

    questions = [
        {
            'type': 'list',
            'name': 'target_model',
            'message': 'Which trained model you want to run? (General Road)',
            'choices':log_files
        }
    ]

    answers = prompt(questions)

    f = open(answers['target_model']+'/log.monitor.csv', 'r')
    _args = json.loads(f.readline().replace('#',''))['args']
    _args['play'] = True
    _args['disable_matlab'] = True

    model_files = sorted(glob(answers['target_model'].replace('.monitor.csv','')+'/*_model.pkl'))
    model_files.sort(key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])

    _args['model_path'] = model_files[-1]

    questions = [
        {
            'type': 'list',
            'name': 'target_model',
            'message': 'Which trained model you want to run? (Bumps)',
            'choices':log_files
        }
    ]

    answers = prompt(questions)

    model_files = sorted(glob(answers['target_model'].replace('.monitor.csv','')+'/*_model.pkl'))
    model_files.sort(key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])

    _args['second_model_path'] = model_files[-1]
    
    _args['prefix'] = "switching_" + args.prefix
    if args.prefix != "":
        _args['prefix'] = args.prefix

    

    args = Namespace(**_args)
    print("Load saved args", args)
    f.close()

    main(args=args)

