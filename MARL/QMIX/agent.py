import numpy as np
import torch
from policy import QMIX


class Agents:
    def __init__(self, args):
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        self.policy = QMIX(args)
        self.last_action = args.last_action
        self.reuse_network = args.reuse_network
        self.save_cycle = args.save_cycle

    def choose_action(self, obs, last_action, agent_num, avail_actions, epsilon):
        inputs = obs.copy()
        avail_actions_idx = np.nonzero(avail_actions)[0]
        agent_id = np.zeros(self.n_agents)
        agent_id[agent_num] = 1.
        if self.last_action:
            inputs = np.hstack((inputs, last_action))
        if self.reuse_network:
            inputs = np.hstack((inputs, agent_id))
        hidden_state = self.policy.eval_hidden[agent_num, :]
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0).cuda()
        avail_actions = torch.tensor(avail_actions, dtype=torch.float32).unsqueeze(0).cuda()
        q_value, self.policy.eval_hidden[agent_num, :] = self.policy.eval_rnn(inputs, hidden_state)
        q_value[avail_actions == 0] = -float('inf')
        if np.random.uniform() < epsilon:
            action = np.random.choice(avail_actions_idx)
        else:
            action = torch.argmax(q_value)
        return action.item()

    def train(self, batch, train_step):
        self.policy.learn(batch, train_step)
        if train_step > 0 and train_step % self.save_cycle == 0:
            self.policy.save_model(train_step // self.save_cycle)

