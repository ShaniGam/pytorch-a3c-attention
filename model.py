import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1, keepdim=True))
    return out


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        if m.bias is not None:
            m.bias.data.fill_(0)


class ActorCritic(torch.nn.Module):
    def __init__(self, num_inputs, action_space):
        super(ActorCritic, self).__init__()
        self.conv1 = nn.Conv2d(num_inputs, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 256, 3, stride=1)

        self.out_hidden_size = 256

        self.lstm = nn.LSTMCell(256, self.out_hidden_size)

        num_outputs = action_space.n
        self.critic_linear = nn.Linear(self.out_hidden_size, 1)
        self.actor_linear = nn.Linear(self.out_hidden_size, num_outputs)

        self.actor_linear.weight.data = normalized_columns_initializer(
            self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = normalized_columns_initializer(
            self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

        self.context_att = nn.Linear(self.out_hidden_size, self.out_hidden_size)
        self.hidden_att = nn.Linear(self.out_hidden_size, self.out_hidden_size, bias=False)
        self.joint_att = nn.Linear(self.out_hidden_size, 1)

        self.apply(weights_init)
        self.softmax = nn.Softmax()
        self.train()

    def forward(self, inputs):
        inputs, (hx, cx) = inputs

        # convolution
        x = F.elu(self.conv1(inputs))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))

        x = x.view(-1, 49, 256)

        context = self.context_att(x.view(-1, 256))
        h = self.hidden_att(hx)
        context = F.tanh(context + h)

        alpha = self.joint_att(context)
        alpha = self.softmax(alpha.view(1, 49))
        context = context.unsqueeze(0)
        alpha = alpha.unsqueeze(2)

        context = torch.bmm(alpha.transpose(1,2), x).squeeze(0)
        hx, cx = self.lstm(context, (hx, cx))
        x = hx
        return self.critic_linear(x), self.actor_linear(x), (hx, cx)
