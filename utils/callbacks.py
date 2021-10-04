import numpy as np
import os
from stable_baselines.results_plotter import load_results, ts2xy
import tensorflow as tf

best_mean_reward, n_steps = -np.inf, 0

# Create log dir
log_dir = "tmp/"
results_dir = "logs/"
benchmark_dir = "benchmarks/"

os.makedirs(log_dir, exist_ok=True)

def tfSummary(tag, val):
    """ Scalar Value Tensorflow Summary
    """
    return tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=val)])

def getBestRewardCallback(args):
    if not os.path.isdir(log_dir + args.prefix):
        os.makedirs(log_dir + args.prefix, exist_ok=True)

    def bestRewardCallback(_locals, _globals, model=None):
        """
        Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
        :param _locals: (dict)
        :param _globals: (dict)
        """
        global n_steps, best_mean_reward
        # Print stats every 1000 calls
        if args.alg == 'ppo2':
            divider = 2
        elif args.alg == 'sac':
            divider = 1
        
        if (n_steps + 1) % divider == 0 and (n_steps + 1) / divider > 1:
        # if True:
            # Evaluate policy training performance
            x, y = ts2xy(load_results(log_dir+args.prefix), 'timesteps')
            if len(x) > 0:
                mean_reward = np.mean(y[-100:])
                # print(mean_reward)
                # print(x[-1], 'timesteps')
                # print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(best_mean_reward, mean_reward))

                # New best model, you could save the agent here
                if mean_reward > best_mean_reward:
                    best_mean_reward = mean_reward
                    # Example for saving best model
                    print("Saving new best model",best_mean_reward)
                    if model is not None:
                        model.save(log_dir + args.prefix + '/' + str(n_steps) +'_best_model.pkl')
                    else:
                        _locals['self'].save(log_dir + args.prefix + '/' + str(n_steps) +'_best_model.pkl')
                elif (n_steps + 1) % (divider*200) == 0:
                    if not os.path.exists(log_dir + args.prefix + '/checkpoints'):
                        os.makedirs(log_dir + args.prefix + '/checkpoints')
                    print("Saving checkpoint",best_mean_reward)
                    if model is not None:
                        model.save(log_dir + args.prefix + '/checkpoints/' + str(n_steps) +'_Check_model.pkl')
                    else:
                        _locals['self'].save(log_dir + args.prefix + '/checkpoints/' + str(n_steps) +'_Check_model.pkl')
                    
                if model is not None:
                    model.save(log_dir + args.prefix + '/model.pkl')
                else:
                    _locals['self'].save(log_dir + args.prefix + '/model.pkl')
       

            # self_ = _locals['self'].env

            # if 'rms_com' in self_.get_attr("info")[0]:
            #     summary = tfSummary('rms/com',self_.get_attr("info")[0]["rms_com"])
            #     _locals['writer'].add_summary(summary, self_.get_attr("episode_cnt")[0]*self_.get_attr("steps_per_episode")[0]+self_.get_attr("step_cnt")[0])

            #     summary = tfSummary('rms/phi',self_.get_attr("info")[0]["rms_phi"])
            #     _locals['writer'].add_summary(summary, self_.get_attr("episode_cnt")[0]*self_.get_attr("steps_per_episode")[0]+self_.get_attr("step_cnt")[0])

            #     summary = tfSummary('rms/theta',self_.get_attr("info")[0]["rms_theta"])
            #     _locals['writer'].add_summary(summary, self_.get_attr("episode_cnt")[0]*self_.get_attr("steps_per_episode")[0]+self_.get_attr("step_cnt")[0])

        n_steps += 1

        return True

    return bestRewardCallback

def rmsLogging(_locals, globals_):
    self_ = _locals['self']

    if self_.done:
        summary = tf.Summary(value=[tf.Summary.Value(tag='rms/com', simple_value=self_.info['rms_com'])])
        _locals['writer'].add_summary(summary, self_.num_timesteps)

        summary = tf.Summary(value=[tf.Summary.Value(tag='rms/phi', simple_value=self_.info['rms_phi'])])
        _locals['writer'].add_summary(summary, self_.num_timesteps)

        summary = tf.Summary(value=[tf.Summary.Value(tag='rms/theta', simple_value=self_.info['rms_theta'])])
        _locals['writer'].add_summary(summary, self_.num_timesteps)
    return True


def logDir():
    return log_dir

def resultsDir():
    return results_dir

def benchmarkDir():
    return benchmark_dir