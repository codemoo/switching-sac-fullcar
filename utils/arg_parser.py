
def arg_parser():
    """
    Create an empty argparse.ArgumentParser.
    """
    import argparse
    return argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

def common_arg_parser():
    """
    Create an argparse.ArgumentParser for run_mujoco.py.
    """
    parser = arg_parser()
    parser.add_argument('--env', help='environment ID', type=str, default='full_car')
    parser.add_argument('--seed', help='RNG seed', type=int, default=None)
    parser.add_argument('--alg', help='Algorithm', type=str, default='sac')
    
    parser.add_argument('--num_timesteps', type=float, default=2e10),
    parser.add_argument('--network', help='network type (mlp, cnn, lstm, cnn_lstm, conv_only)', default='mlp_big')
    parser.add_argument('--num_env', help='Number of environment copies being run in parallel. When not specified, set to number of cpus for Atari, and to 1 for Mujoco', default=None, type=int)
    parser.add_argument('--save_path', help='Path to save trained model to', default=None, type=str)
    parser.add_argument('--play', default=False, action='store_true')

    parser.add_argument('--prefix', default='', type=str)
    parser.add_argument('--model_path', help='Path to load trained model to', default=None, type=str)
    parser.add_argument('--second_model_path', help='Path to load trained model to', default=None, type=str)
    parser.add_argument('--road_type', help='', default='only_rough', type=str)
    parser.add_argument('--reward', help='', default='com_roll_pitch', type=str)
    parser.add_argument('--action_type', help='', default='damping_force', type=str)
    parser.add_argument('--action_cont', help='Action type : Cont or disc', type=str, default='continuous')
    parser.add_argument('--action_size', help='number of action', default=4, type=int)

    parser.add_argument('--road_cls', help='none/ground_truth/neural_net', default='none', type=str)
    
    parser.add_argument('--lstm_on', help='LSTM', default=False, type=bool)
    parser.add_argument('--network_size', help='plot', default=64, type=int)
    parser.add_argument('--layer_size', help='plot', default=2, type=int)
    parser.add_argument('--delay_size', help='plot', default=0, type=int)
    parser.add_argument('--regularize', help='plot', default=0, type=int)
    parser.add_argument('--sat_limit', help ='sat_limit', default='1500', type=int)
    parser.add_argument('--symmetric', help ='Symmetric', default=False)
    parser.add_argument('--imitation', help ='Symmetric', default=False)
    parser.add_argument('--half_like', help ='Symmetric', default=False)

    parser.add_argument('--bump_switching', help ='switch model when bump', default=False, type=bool)

    parser.add_argument('--state_reconstruction', help ='kf, nn', default="None")
    parser.add_argument('--state_recon_window_size', help='state_recon_window_size Size', default=8, type=int)

    parser.add_argument('--window_size', help='Window Size', default=2, type=int)
    parser.add_argument('--sampling_freq', help ='Sampling Freq.', default=100, type=int)
    parser.add_argument('--noise_on', help = 'adding sensor noise', default = False, type=bool)
    parser.add_argument('--actuator_dynamics', help = 'AD', default = False, type = bool)

    parser.add_argument('--disable_matlab', help = 'AD', default = True, type = bool)
    parser.add_argument('--road_data', help='', default='damping_force', type=str)

    parser.add_argument('--add_road_index', help = 'AD', default = False, type = bool)
    parser.add_argument('--separate_training', help = 'AD', default = False, type = bool)
    parser.add_argument('--separate_training_target', help='', default='general_road', type=str)

    parser.add_argument('--quarter_car_position', help='', default='', type=str)

    parser.add_argument('--psd_window_size', help='', default=50, type=int)
    parser.add_argument('--psd_weight', help='', default='', type=str)
    parser.add_argument('--psd_ratio', help='', default=10, type=int)

    parser.add_argument('--window_sm', help='', default=20, type=int)
    parser.add_argument('--window_out_big', help='', default=200, type=int)
    parser.add_argument('--dewel_time', help='', default=200, type=int)
    parser.add_argument('--in_acc_threshold_sm', help='', default=80, type=int)
    parser.add_argument('--out_acc_threshold', help='', default=10, type=int)

    return parser

def parseLayersFromArgs(args):
    layers = []
    for l in range(args.layer_size):
        layers.append(args.network_size)

    return layers