import torch
import random
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


# DQN with tanh activation function
class DQN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x


# Replay Buffer
class ReplayBuffer:
    def __init__(self, buffer_size, state_dim, batch_size):
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
    def __init__(self, model, buffer, opt, state_size, action_size, gamma):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.state_size = state_size
        self.action_size = action_size
        self.model = model
        self.optimizer = opt
        self.criterion = nn.MSELoss()
        self.buffer = buffer

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

        q_value = self.model(state).gather(1, action).squeeze(1)
        next_q_value = self.model(next_state).max(1)[0]
        expected_q_value = reward + self.gamma * next_q_value * (1 - done)
        loss = self.criterion(q_value, expected_q_value.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def get_action(self, state, epsilon):
        if epsilon != 0 and np.random.rand() <= epsilon:
            return np.random.randint(state.shape[0], self.action_size)
        else:
            state = torch.FloatTensor(state).to(self.device)
            q_value = self.model(state)
            return q_value.max(1)[1].detach().numpy() if len(
                q_value.shape) > 1 else q_value.max(0)[1].detach().numpy()

    def get_policy_f(self):
        from functools import partial
        return partial(self.get_action, epsilon=0.0)

    def get_state_value(self, state):
        state = torch.FloatTensor(state).to(self.device)
        q_value = self.model(state)
        # get the max value of the state and flatten it
        return q_value.max(1)[0].detach().numpy()

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        print("Loading model from {}".format(path))
        self.model.load_state_dict(torch.load(path))


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
