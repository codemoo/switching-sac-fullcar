import sys

from stable_baselines.sac.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.bench import Monitor

from stable_baselines import SAC

from envs.full_car_env import FullCar

from utils.callbacks import getBestRewardCallback, logDir
from utils.arg_parser import common_arg_parser, parseLayersFromArgs

import os

def main(args):
    arg_parser = common_arg_parser()
    args, unknown_args = arg_parser.parse_known_args(args)

    env = FullCar(args=args)
    
    env = Monitor(env, logDir()+args.prefix+"/log", allow_early_resets=True)

    layers = parseLayersFromArgs(args=args) # default [32, 32]
    bestRewardCallback = getBestRewardCallback(args)

    policy_kwargs = dict(layers=layers)

    
    env = DummyVecEnv([lambda: env])

    model = SAC(MlpPolicy, env, verbose=1, tensorboard_log=os.path.join("tensorboard_"+args.env,args.prefix), policy_kwargs=policy_kwargs)

    model.learn(total_timesteps=20000*2000, log_interval=100, callback=bestRewardCallback)

if __name__ == '__main__':
    main(sys.argv)

