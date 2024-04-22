import numpy as np
import os
import matplotlib.pyplot as plt
from agent import Agents
from tqdm import tqdm
from rollout import RolloutWorker
from replay_buffer import ReplayBuffer


class Runner:
    def __init__(self, env, args):
        self.env = env
        self.evaluate_epoch = args.evaluate_epoch
        self.agents = Agents(args)
        self.rolloutWorker = RolloutWorker(env, self.agents, args)
        self.replay_buffer = ReplayBuffer(args)
        self.n_steps = args.n_steps
        self.n_episodes = args.n_episodes
        self.train_steps = args.train_steps
        self.batch_size = args.batch_size
        self.evaluate_cycle = args.evaluate_cycle
        self.win_rates = []
        self.episode_rewards = []
        self.evaluate_win_rates = []
        self.evaluate_episode_rewards = []

        self.save_path = args.result_dir + '/' + args.map + '/' + 'evaluate'
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def run(self):
        time_steps, train_steps, evaluate_steps = 0, 0, 0
        with tqdm(total=self.n_steps, desc='Iteration %i' % time_steps) as pbar:
            while time_steps < self.n_steps:
                if time_steps // self.evaluate_cycle > evaluate_steps:
                    win_rate, episode_reward = self.evaluate()
                    self.evaluate_win_rates.append(win_rate)
                    self.evaluate_episode_rewards.append(episode_reward)
                    self.plt()
                    evaluate_steps += 1
                for episode_idx in range(self.n_episodes):
                    episode, episode_reward, win_tag, steps = self.rolloutWorker.generate_episode(episode_idx)
                    self.replay_buffer.store_episode(episode)
                    time_steps += steps
                    self.win_rates.append(win_tag)
                    self.episode_rewards.append(episode_reward)
                    pbar.update(steps)
                for train_step in range(self.train_steps):
                    mini_batch = self.replay_buffer.sample(min(self.replay_buffer.len(), self.batch_size))
                    self.agents.train(mini_batch, train_steps)
                    train_steps += 1
                pbar.set_postfix({
                    'return': '%.3f' % np.mean(self.episode_rewards[-10:]),
                    'win_rate': '%.3f' % np.mean(self.win_rates[-10:]),
                    'epsilon': '%.6f' % self.rolloutWorker.epsilon,
                    'train_steps': '%d' % train_steps
                })
        win_rate, episode_reward = self.evaluate()
        print('win_rate is ', win_rate)
        self.win_rates.append(win_rate)
        self.episode_rewards.append(episode_reward)
        self.plt()

    def evaluate(self):
        win_number = 0
        episode_rewards = 0
        for epoch in range(self.evaluate_epoch):
            _, episode_reward, win_tag, _ = self.rolloutWorker.generate_episode(epoch, evaluate=True)
            episode_rewards += episode_reward
            if win_tag:
                win_number += 1
        return win_number / self.evaluate_epoch, episode_rewards / self.evaluate_epoch

    def plt(self):
        plt.figure()
        plt.cla()
        plt.plot(range(len(self.win_rates)), self.win_rates)
        plt.xlabel('step*{}'.format(self.evaluate_cycle))
        plt.ylabel('win_rates')
        plt.savefig(self.save_path + '/win_rates.png', format='png')
        plt.close()

        plt.figure()
        plt.cla()
        plt.plot(range(len(self.evaluate_win_rates)), self.evaluate_win_rates)
        plt.xlabel('step*{}'.format(self.evaluate_cycle))
        plt.ylabel('evaluate_win_rates')
        plt.savefig(self.save_path + '/evaluate_win_rates.png', format='png')
        plt.close()

        plt.figure()
        plt.cla()
        plt.plot(range(len(self.episode_rewards)), self.episode_rewards)
        plt.xlabel('step*{}'.format(self.evaluate_cycle))
        plt.ylabel('episode_rewards')
        plt.savefig(self.save_path + '/episode_rewards.png', format='png')
        plt.close()

        plt.figure()
        plt.cla()
        plt.plot(range(len(self.evaluate_episode_rewards)), self.evaluate_episode_rewards)
        plt.xlabel('step*{}'.format(self.evaluate_cycle))
        plt.ylabel('evaluate_episode_rewards')
        plt.savefig(self.save_path + '/evaluate_episode_rewards.png', format='png')
        plt.close()

        np.save(self.save_path + '/win_rates', self.win_rates)
        np.save(self.save_path + '/evaluate_win_rates', self.evaluate_win_rates)
        np.save(self.save_path + '/episode_rewards', self.episode_rewards)
        np.save(self.save_path + '/evaluate_episode_rewards', self.evaluate_episode_rewards)
        self.env.save_replay()
