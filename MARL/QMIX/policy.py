import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from net import QMixNet, RNN
import torch.nn.functional as F


class QMIX:
    def __init__(self, args):
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        self.rnn_hidden_dim = args.rnn_hidden_dim
        self.last_action = args.last_action
        self.reuse_network = args.reuse_network
        self.gamma = args.gamma
        self.model_dir = args.model_dir
        self.target_update_cycle = args.target_update_cycle
        self.grad_norm_clip = args.grad_norm_clip
        self.save_path = args.result_dir + '/' + args.map + '/' + 'evaluate'
        input_shape = self.obs_shape
        # 根据参数决定RNN的输入维度
        if args.last_action:
            input_shape += self.n_actions
        if args.reuse_network:
            input_shape += self.n_agents
        # 网络
        self.eval_rnn = RNN(input_shape, args).cuda()
        self.target_rnn = RNN(input_shape, args).cuda()
        self.eval_qmix_net = QMixNet(args).cuda()
        self.target_qmix_net = QMixNet(args).cuda()
        self.losses = []
        if args.load_model:
            if os.path.exists('qmix_model/' + args.map):
                path_rnn = 'qmix_model/' + args.map + '/rnn_net_params.pkl'
                path_qmix = 'qmix_model/' + args.map + '/qmix_net_params.pkl'
                map_location = 'cuda:0'
                self.eval_rnn.load_state_dict(torch.load(path_rnn, map_location=map_location))
                self.eval_qmix_net.load_state_dict(torch.load(path_qmix, map_location=map_location))
                print('Successfully load the model ' + args.map)
            else:
                raise Exception('No model')
        self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
        self.target_qmix_net.load_state_dict(self.eval_qmix_net.state_dict())
        self.eval_parameters = list(self.eval_qmix_net.parameters()) + list(self.eval_rnn.parameters())
        self.optimizer = torch.optim.RMSprop(self.eval_parameters, lr=args.lr)
        #RNN中的h_{t-1}
        self.eval_hidden = None
        self.target_hidden = None

    def learn(self, batch, train_step):
        episode_num = len(batch)
        self.optimizer.zero_grad()
        for i_episode in range(episode_num):
            self.init_hidden()
            s = torch.tensor(batch[i_episode]['s'], dtype=torch.float32)[0].cuda()  # (episode_len, state_shape)
            s_next = torch.tensor(batch[i_episode]['s_next'], dtype=torch.float32)[0].cuda()
            obs = torch.tensor(batch[i_episode]['o'], dtype=torch.float32)[0]  # (episode_len, n_agents, obs_shape)
            obs_next = torch.tensor(batch[i_episode]['o_next'], dtype=torch.float32)[0]
            u = torch.tensor(batch[i_episode]['u'], dtype=torch.int64)[0].cuda()  # (episode_len, n_agents, 1)
            last_u = torch.tensor(batch[i_episode]['last_u'], dtype=torch.int64)[0]  # (episode_len, n_agents, n_actions)
            avail_u_next = torch.tensor(batch[i_episode]['avail_u_next'], dtype=torch.int64)[0]
            u_one_hot = torch.tensor(batch[i_episode]['u_one_hot'], dtype=torch.int64)[0]
            r = torch.tensor(batch[i_episode]['r'], dtype=torch.float32)[0].cuda()  # (episode_len, 1)
            terminated = torch.tensor(batch[i_episode]['terminated'], dtype=torch.float32)[0].cuda()  # (episode_len, 1)
            inputs, inputs_next = obs, obs_next
            if self.last_action:
                inputs = torch.cat((inputs, last_u), dim=2)
                inputs_next = torch.cat((inputs_next, u_one_hot), dim=2)
            if self.reuse_network:
                agent_id = torch.eye(self.n_agents).unsqueeze(0).expand(len(inputs), -1, -1)
                inputs = torch.cat((inputs, agent_id), dim=2)
                inputs_next = torch.cat((inputs_next, agent_id), dim=2)
            inputs = inputs.cuda()  # (episode_len, n_agents, input_dim)
            inputs_next = inputs_next.cuda()
            q_evals, q_targets = torch.tensor([]).cuda(), torch.tensor([]).cuda()
            for i, i_next, a, avail_a_next in zip(inputs, inputs_next, u, avail_u_next):  # (n_agents, input_dim), (n_agents, 1)
                q_eval, self.eval_hidden = self.eval_rnn(i, self.eval_hidden)  # (n_agents, n_actions)
                q_target, self.target_hidden = self.target_rnn(i_next, self.target_hidden)
                q_eval = q_eval.gather(1, a).squeeze(1)  # (n_agents)
                q_target[avail_a_next == 0.0] = -float('inf')
                q_target = q_target.max(dim=1)[0]  # (n_agents)
                q_evals = torch.hstack((q_evals, q_eval))
                q_targets = torch.hstack((q_targets, q_target))

            q_evals = q_evals.reshape(-1, self.n_agents)
            q_targets = q_targets.reshape(-1, self.n_agents)
            # 由于RNN网络结构的缘故，必须以整个episode按时间步依次获得hidden_state，q_total以episode计算是为了利用tensor内置的矩阵计算
            q_total_eval = self.eval_qmix_net(q_evals, s).reshape(-1)
            q_total_target = self.target_qmix_net(q_targets, s_next).squeeze(2)
            targets = r + self.gamma * q_total_target * (1 - terminated)
            targets = targets.reshape(-1)
            loss = torch.mean(F.mse_loss(q_total_eval, targets))
            self.losses.append(loss.item())
            loss.backward()

        # torch.nn.utils.clip_grad_norm_(self.eval_parameters, self.grad_norm_clip)
        self.optimizer.step()
        if train_step > 0 and train_step % self.target_update_cycle == 0:
            self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
            self.target_qmix_net.load_state_dict(self.eval_qmix_net.state_dict())
            plt.figure()
            plt.plot(range(len(self.losses)), self.losses)
            plt.xlabel('step')
            plt.ylabel('loss')
            plt.savefig(self.save_path + '/loss.png', format='png')
            np.save(self.save_path + '/loss', self.losses)
            plt.close()

    def init_hidden(self):
        self.eval_hidden = torch.zeros((self.n_agents, self.rnn_hidden_dim)).cuda()
        self.target_hidden = torch.zeros((self.n_agents, self.rnn_hidden_dim)).cuda()

    def save_model(self, num):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        torch.save(self.eval_qmix_net.state_dict(), self.model_dir + '/' + str(num) + '_qmix_net_params.pkl')
        torch.save(self.eval_rnn.state_dict(), self.model_dir + '/' + str(num) + '_rnn_net_params.pkl')
