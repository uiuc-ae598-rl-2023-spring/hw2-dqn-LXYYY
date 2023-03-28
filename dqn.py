import torch
import random
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import re
import os


# DQN with tanh activation function
def get_activ(a):
    if a == 't':
        return nn.Tanh(), 'nn.Tanh()'
    elif a == 'r':
        return nn.ReLU(), 'nn.ReLU()'
    elif a == 'l':
        return nn.LeakyReLU(), 'nn.LeakyReLU()'
    elif a == 'e':
        return nn.ELU(), 'nn.ELU()'
    elif a == 's':
        return nn.Sigmoid(), 'nn.Sigmoid()'
    else:
        raise ValueError('Unknown activation function: ' + a)


def parse_string(hs: str):
    # split hs into a list of ints and letters
    h = re.findall(r'\d+|\D+', hs)
    # check all odd id are letter
    assert all([h[i].isalpha() for i in range(0, len(h), 2)])
    # check all even id are int
    assert all([h[i].isdigit() for i in range(1, len(h), 2)])
    activ = [h[i] for i in range(0, len(h), 2)]
    sizes = [int(h[i]) for i in range(1, len(h), 2)]
    assert len(activ) == len(sizes)
    return activ, sizes


class DQN(nn.Module):

    def __init__(self, input_dim, hs: str, output_dim):
        super(DQN, self).__init__()
        # parse hs
        activ, sizes = parse_string(hs)
        print('Build a DQN with hidden layers: ' + hs)
        self.layers = nn.ModuleList([nn.Linear(input_dim, sizes[0])])
        print('Linear(' + str(input_dim) + ', ' + str(sizes[0]) + ')')

        for i in range(0, len(sizes)):
            activ_layer, name = get_activ(activ[i])
            self.layers.append(activ_layer)
            print(name)
            s_in = sizes[i]
            s_out = sizes[i + 1] if i + 1 < len(sizes) else output_dim
            self.layers.append(nn.Linear(s_in, s_out))
            print('Linear(' + str(s_in) + ', ' + str(s_out) + ')')

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# Replay Buffer
class ReplayBuffer:
    def __init__(self, buffer_size, state_dim):
        self.buffer_size = buffer_size
        self.ptr = 0
        self.cur_size = 0
        self.state_dim = state_dim
        self.states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.actions = np.zeros(buffer_size, dtype=np.int32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.next_states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.terminals = np.zeros(buffer_size, dtype=np.uint8)

    def add(self, state, action, reward, next_state, terminal):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.terminals[self.ptr] = terminal
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.cur_size = min(self.cur_size + 1, self.buffer_size)

    def sample(self, batch_size):
        indices = np.random.choice(
            self.cur_size, batch_size, replace=False)
        return self.states[indices], self.actions[indices], self.rewards[indices], self.next_states[indices], \
            self.terminals[indices]

    def __len__(self):
        return self.cur_size


class Agent:
    def __init__(self, target_network, policy_network, buffer, opt, state_size, action_size, target_update, gamma):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.state_size = state_size
        self.action_size = action_size
        self.target_network = target_network
        self.policy_network = policy_network
        self.optimizer = opt
        self.criterion = nn.MSELoss()
        self.buffer = buffer
        self.target_update = target_update
        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.step = 0

    def update(self, batch_size):
        state, action, reward, next_state, done = self.buffer.sample(
            batch_size)
        state = torch.FloatTensor(state).to(
            self.device).reshape(batch_size, -1)
        next_state = torch.FloatTensor(next_state).to(
            self.device).reshape(batch_size, -1)
        action = torch.LongTensor(action).to(
            self.device).reshape(batch_size, -1)
        reward = torch.FloatTensor(reward).to(self.device).reshape(batch_size)
        done = torch.FloatTensor(done).to(self.device).reshape(batch_size)

        q_value = self.policy_network(state).gather(1, action).squeeze(1)
        next_q_value = self.target_network(next_state).max(1)[0]
        expected_q_value = reward + self.gamma * next_q_value * (1 - done)
        loss = self.criterion(q_value, expected_q_value.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.step += 1
        if self.step % self.target_update == 0:
            self.target_network.load_state_dict(self.policy_network.state_dict())
            self.step = 0

    def update_target(self):
        self.target_network.load_state_dict(self.policy_network.state_dict())

    def get_action(self, state, epsilon):
        if epsilon != 0 and np.random.rand() <= epsilon:
            return np.random.randint(self.action_size)
        else:
            state = torch.FloatTensor(state).to(self.device)
            q_value = self.target_network(state)
            return q_value.max(1)[1].detach().numpy() if len(
                q_value.shape) > 1 else q_value.max(0)[1].detach().numpy()

    def get_policy_f(self):
        from functools import partial
        return partial(self.get_action, epsilon=0.0)

    def get_state_value(self, state):
        state = torch.FloatTensor(state).to(self.device)
        q_value = self.target_network(state)
        # get the max value of the state and flatten it
        return q_value.max(1)[0].detach().numpy()

    def save_model(self, path):
        torch.save(self.target_network.state_dict(), path)

    def load_model(self, path):
        # check path exists
        if os.path.isfile(path):
            print("Loading model from {}".format(path))
            self.target_network.load_state_dict(torch.load(path))
        else:
            print("No model found at {}".format(path))


def train(env, agent, num_episodes, batch_size, epsilon, epsilon_decay, epsilon_min, render=False):
    scores = []
    for e in range(num_episodes):
        state = env.reset()
        score = 0
        while True:
            if render:
                env.render()
            action = agent.get_action(state, epsilon)
            next_state, reward, done = env.step(action)
            agent.buffer.add(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if len(agent.buffer) >= batch_size:
                agent.update(batch_size)
            if done:
                break
        epsilon = max(epsilon * epsilon_decay, epsilon_min)
        scores.append(score)
        print("episode: {}, score: {:.1f}, memory length: {}, epsilon: {:.1f}".format(
            e, score, len(agent.buffer), epsilon))
    return scores
