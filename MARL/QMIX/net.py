import torch
import torch.nn.functional as F


class RNN(torch.nn.Module):
    # input_dim = obs_shape + n_actions + n_agents
    def __init__(self, input_dim, Args):
        super(RNN, self).__init__()
        self.hidden_dim = Args.rnn_hidden_dim
        self.fc1 = torch.nn.Linear(input_dim, self.hidden_dim)
        self.rnn = torch.nn.GRUCell(self.hidden_dim, self.hidden_dim)
        self.fc2 = torch.nn.Linear(self.hidden_dim, Args.n_actions)

    def forward(self, x, hidden_state):
        x = F.relu(self.fc1(x))
        h_in = hidden_state.reshape(-1, self.hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h.cuda()


class QMixNet(torch.nn.Module):
    def __init__(self, Args):
        super(QMixNet, self).__init__()
        self.n_agents = Args.n_agents
        self.state_shape = Args.state_shape
        self.qmix_hidden_dim = Args.qmix_hidden_dim
        # 权重网络
        if Args.two_hyper_layers:
            # hyper_w1需要是一个矩阵，维度为(n_agents, qmix_hidden_dim), 而pytorch只能输出一个向量，故先输出向量在转化为矩阵
            self.hyper_w1 = torch.nn.Sequential(torch.nn.Linear(Args.state_shape, Args.hyper_hidden_dim),
                                                torch.nn.ReLU(),
                                                torch.nn.Linear(Args.hyper_hidden_dim, Args.n_agents * Args.qmix_hidden_dim))
            self.hyper_w2 = torch.nn.Sequential(torch.nn.Linear(Args.state_shape, Args.hyper_hidden_dim),
                                                torch.nn.ReLU(),
                                                torch.nn.Linear(Args.hyper_hidden_dim, Args.qmix_hidden_dim))
        else:
            self.hyper_w1 = torch.nn.Linear(Args.state_shape, Args.n_agents * Args.qmix_hidden_dim)
            self.hyper_w2 = torch.nn.Linear(Args.state_shape, Args.qmix_hidden_dim)

        # 偏置网络
        self.hyper_b1 = torch.nn.Linear(Args.state_shape, Args.qmix_hidden_dim)
        self.hyper_b2 = torch.nn.Sequential(torch.nn.Linear(Args.state_shape, Args.qmix_hidden_dim),
                                            torch.nn.ReLU(),
                                            torch.nn.Linear(Args.qmix_hidden_dim, 1))

    def forward(self, q_values, states):
        # (episode_len, n_agents)
        # (episode_len, state_shape)
        episode_num = q_values.size(0)
        q_values = q_values.view(-1, 1, self.n_agents)  # (max_episode_len, 1, n_agents)
        states = states.reshape(-1, self.state_shape)  # (episode_len, state_shape)

        w1 = torch.abs(self.hyper_w1(states))  # (episode_len, n_agents * qmix_hidden_dim)
        w1 = w1.view(-1, self.n_agents, self.qmix_hidden_dim)  # (episode_len, n_agents, qmix_hidden_dim)
        b1 = self.hyper_b1(states)  # (episode_len, qmix_hidden_dim)
        b1 = b1.view(-1, 1, self.qmix_hidden_dim)  # (episode_len, 1, qmix_hidden_dim)
        hidden = F.elu(torch.bmm(q_values, w1) + b1)  # (episode_len, 1, qmix_hidden_dim)

        w2 = torch.abs(self.hyper_w2(states))  # (episode_len, qmix_hidden_dim)
        w2 = w2.view(-1, self.qmix_hidden_dim, 1)  # (episode_len, qmix_hidden_dim, 1)
        b2 = self.hyper_b2(states)  # (episode_len, 1)
        b2 = b2.view(-1, 1, 1)  # ((episode_len, 1, 1)

        q_total = torch.bmm(hidden, w2) + b2  # (episode_len, 1, 1)
        q_total = q_total.view(episode_num, -1, 1)  # (1, episode_len, 1)
        return q_total
