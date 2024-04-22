import random
import numpy as np
import torch

from smac.env.starcraft2.starcraft2 import StarCraft2Env
import argparse
from runner import Runner


if __name__ == '__main__':
    args = argparse.ArgumentParser().parse_args()
    args.map = '3m'
    args.step_mul = 8  # how many steps to make an action
    args.difficulty = '7'  # the difficulty of the game
    args.game_version = 'latest'
    args.replay_dir = 'replay/'
    args.result_dir = 'results/'
    args.model_dir = 'results/' + args.map + '/model'
    args.two_hyper_layers = False
    args.rnn_hidden_dim = 64
    args.hyper_hidden_dim = 64
    args.qmix_hidden_dim = 32
    args.last_action = True  # whether to use the last action to choose action
    args.reuse_network = True  # whether to use one network for all agents
    args.load_model = False  # whether to load the pretrained model
    args.lr = 5e-4
    args.n_steps = 1000000
    args.n_episodes = 2
    args.evaluate_cycle = 5000
    args.evaluate_epoch = 16
    args.save_cycle = 500
    args.buffer_size = 5000
    args.batch_size = 32
    args.train_steps = 2
    args.gamma = 0.99
    args.epsilon = 1.
    args.min_epsilon = 0.05
    args.anneal_steps = 50000  # epsilon 衰减步数
    args.target_update_cycle = 200
    args.grad_norm_clip = 10  # prevent gradient explosion
    args.epsilon_anneal_way = 'step'
    if args.epsilon_anneal_way not in ['step', 'episode']:
        raise Exception('wrong epsilon-anneal-way')
    args.anneal_epsilon = (args.epsilon - args.min_epsilon) / args.anneal_steps
    map_name = ['3m', '2s3z', '1c3s5z', '5m_vs_6m', '8m', '3s_vs_5z']
    for i in range(6):
        args.map = map_name[i]
        env = StarCraft2Env(map_name=args.map, step_mul=args.step_mul, difficulty=args.difficulty,
                            game_version=args.game_version, replay_dir=args.replay_dir, seed=0)
        env_info = env.get_env_info()
        args.n_actions = env_info['n_actions']
        args.n_agents = env_info['n_agents']
        args.state_shape = env_info['state_shape']
        args.obs_shape = env_info['obs_shape']
        args.episode_limit = env_info['episode_limit']
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        runner = Runner(env, args)
        runner.run()
        # env.reset()
        # print(env.get_avail_actions())
        # i = env.reset()
        env.close()
