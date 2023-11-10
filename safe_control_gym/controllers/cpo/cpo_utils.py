import torch

LOG_STD_MAX = 2
LOG_STD_MIN = -4
EPS = 1e-8


def initWeights(m):
    if isinstance(m, torch.nn.Linear):
        m.weight.data.normal_(0, 0.01)
        m.bias.data.normal_(0, 0.01)


def initWeights2(m):
    if isinstance(m, torch.nn.Linear):
        m.weight.data.normal_(0, 0.01)
        m.bias.data.normal_(-1.0, 0.01)


class CPOPolicy(torch.nn.Module):
    def __init__(self, state_dim, action_dim, hidden1, hidden2):
        super(CPOPolicy, self).__init__()

        self.fc1 = torch.nn.Linear(state_dim, hidden1)
        self.fc2 = torch.nn.Linear(hidden1, hidden2)
        self.act_fn = torch.relu
        self.output_act_fn = torch.sigmoid

        self.fc_mean = torch.nn.Linear(hidden2, action_dim)
        self.fc_log_std = torch.nn.Linear(hidden2, action_dim)

    def forward(self, x):
        x = self.act_fn(self.fc1(x))
        x = self.act_fn(self.fc2(x))
        mean = self.output_act_fn(self.fc_mean(x))
        log_std = self.fc_log_std(x)

        log_std = torch.clamp(log_std, min=LOG_STD_MIN, max=LOG_STD_MAX)
        std = torch.exp(log_std)
        return mean, log_std, std

    def initialize(self):
        for m_idx, module in enumerate(self.children()):
            if m_idx != 3:
                module.apply(initWeights)
            else:
                module.apply(initWeights2)


class CPOValue(torch.nn.Module):
    def __init__(self, state_dim, hidden1, hidden2):
        super(CPOValue, self).__init__()

        self.fc1 = torch.nn.Linear(state_dim, hidden1)
        self.fc2 = torch.nn.Linear(hidden1, hidden2)
        self.fc3 = torch.nn.Linear(hidden2, 1)
        self.act_fn = torch.relu

    def forward(self, x):
        x = self.act_fn(self.fc1(x))
        x = self.act_fn(self.fc2(x))
        x = self.fc3(x)
        x = torch.reshape(x, (-1,))
        return x

    def initialize(self):
        self.apply(initWeights)
