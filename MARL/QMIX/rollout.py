# 在强化学习中，rollout指的是在训练过程中，智能体根据当前的策略在环境中进行一系列的交互步骤，模拟并收集样本数据的过程。

import numpy as np
import torch


class RolloutWorker:
    def __init__(self, env, agents, args):
        self.env = env
        self.agents = agents
        self.episode_limit = args.episode_limit
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        self.epsilon = args.epsilon
        self.anneal_epsilon = args.anneal_epsilon
        self.min_epsilon = args.min_epsilon
        self.epsilon_anneal_way = args.epsilon_anneal_way
        self.evaluate_epoch = args.evaluate_epoch
        self.replay_dir = args.replay_dir

    @torch.no_grad()
    def generate_episode(self, episode_num=None, evaluate=False):
        # if self.replay_dir != '' and evaluate and episode_num == 0:
        #     self.env.close()
        o, u, r, s, avail_u, last_u, u_ont_hot, terminate = [], [], [], [], [], [], [], []
        self.env.reset()
        terminated = False
        win_tag = False
        step = 0
        episode_reward = 0
        last_action = np.zeros((self.n_agents, self.n_actions))
        last_u.append(last_action)
        self.agents.policy.init_hidden()
        epsilon = 0 if evaluate else self.epsilon
        if self.epsilon_anneal_way == 'episode':
            epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon
        while not terminated and step < self.episode_limit:
            obs = self.env.get_obs()
            state = self.env.get_state()
            actions, avail_actions, actions_one_hot = [], [], []
            for agent_id in range(self.n_agents):
                avail_action = self.env.get_avail_agent_actions(agent_id)
                action = self.agents.choose_action(obs[agent_id], last_action[agent_id], agent_id, avail_action, epsilon)
                action_ont_hot = np.zeros(self.n_actions)

                action_ont_hot[action] = 1
                actions.append(action)
                actions_one_hot.append(action_ont_hot)
                avail_actions.append(avail_action)
                last_action[agent_id] = action_ont_hot
            reward, terminated, info = self.env.step(actions)
            win_tag = True if terminated and 'battle_won' in info and info['battle_won'] else False
            o.append(obs)
            s.append(state)
            u.append(np.reshape(actions, [self.n_agents, 1]))
            u_ont_hot.append(actions_one_hot)
            avail_u.append(avail_actions)
            last_u.append(last_action)
            r.append([reward])
            terminate.append([terminated])
            episode_reward += reward
            step += 1
            if self.epsilon_anneal_way == 'step':
                epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon
        # take action 后环境状态
        obs = self.env.get_obs()
        state = self.env.get_state()
        o.append(obs)
        s.append(state)
        o_next = o[1:]
        s_next = s[1:]
        o = o[:-1]
        s = s[:-1]
        last_u = last_u[:-1]
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_action = self.env.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_action)
        avail_u.append(avail_actions)
        avail_u_next = avail_u[1:]
        avail_u = avail_u[:-1]

        episode = dict(o=o.copy(), s=s.copy(), u=u.copy(), r=r.copy(), avail_u=avail_u.copy(), o_next=o_next.copy(),
                       s_next=s_next.copy(), avail_u_next=avail_u_next.copy(), u_one_hot=u_ont_hot.copy(),
                       terminated=terminate.copy(), last_u=last_u.copy())
        for key in episode.keys():
            episode[key] = np.array([episode[key]])
        if not evaluate:
            self.epsilon = epsilon
        if evaluate and episode_num == self.evaluate_epoch - 1 and self.replay_dir != '':
            self.env.save_replay()
            # self.env.close()
        return episode, episode_reward, win_tag, step
