import random
import numpy as np
import matplotlib

# Set the backend to 'agg' to turn off plot windows
matplotlib.use('agg')

import matplotlib.pyplot as plt
import discreteaction_pendulum
import sys
import argparse
import torch.optim as optim

import dqn
import plot
from utils import *


# Main function with arguments
def main():
    # arg parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', action='store_true', help='load checkpoint')
    parser.add_argument('--save', action='store_true', help='save checkpoint')
    parser.add_argument('--train', action='store_true', help='train model')
    parser.add_argument('--exp', type=str, help='experiment name')
    parser.add_argument('--config', type=str, help='config file name')

    load_ckp = parser.parse_args().load if parser.parse_args().load else False
    train = parser.parse_args().train if parser.parse_args().train else False
    config = parser.parse_args().config if parser.parse_args().config else 'ablation.json'
    # get exps from config file
    networks = read_json_file(config)['networks']
    episodes = read_json_file(config)['episodes']
    display = True if len(networks) == 1 else False

    print('Found {} experiments'.format(len(networks)))

    for i, exp in enumerate(networks):

        print('Experiment {} / {}: {}'.format(i + 1, len(networks), exp))
        ckp = 'data/' + exp + '.pth'

        # Create environment
        #
        #   By default, the action space (tau) is discretized with 31 grid points.
        #
        #   You can change the number of grid points as follows (for example):
        #
        #       env = discrete_pendulum.Pendulum(num_actions=21)
        #
        #   Note that there will only be a grid point at "0" if the number of grid
        #   points is odd.
        #
        #   How does performance vary with the number of grid points? What about
        #   computation time?
        env = discreteaction_pendulum.Pendulum()

        h, lr, e, ed, em, tu, bs, rs = get_parameters_from_description(exp)
        gamma = 0.95

        target_network = dqn.DQN(env.num_states, h, env.num_actions)
        policy_network = dqn.DQN(env.num_states, h, env.num_actions)
        buffer = dqn.ReplayBuffer(
            buffer_size=rs, state_dim=env.num_states)
        opt = optim.Adam(policy_network.parameters(), lr=lr)

        agent = dqn.Agent(target_network, policy_network, buffer, opt, env.num_states,
                          env.num_states, tu, gamma)

        scores = None
        if load_ckp:
            agent.load_model(ckp)
        if train:
            # Train agent
            scores = dqn.train(env, agent, num_episodes=episodes, batch_size=bs,
                               epsilon=e, epsilon_decay=ed, epsilon_min=em, render=False)

            # save model
            agent.save_model(ckp)

        ######################################
        #
        #   EXAMPLE OF CREATING A VIDEO
        #

        policy = agent.get_policy_f()

        # Simulate an episode and save the result as an animated gif
        env.video(policy=policy,
                  filename='figures/test_discreteaction_pendulum_' + exp + '.gif')

        # Plot actions of a sample trajectory
        s = env.reset()
        done = False
        na = []
        while not done:
            a = policy(s)
            s, r, done = env.step(a)
            na.append(a)

        plt.figure('Actions')
        plt.plot(na)

        if scores is not None:
            # save scores
            np.save('data/scores_' + exp + '.npy', scores)
        #
        #     plt.figure()
        #     plt.ylim(0, 100)
        #     plot.plot_learning_curves(
        #         [scores], [exp], save=True)

        # plot state-value function
        p = plot.Plot(env, "pendumlum", "dqn")
        ntheta = np.linspace(-np.pi, np.pi, 200)
        nthetadot = np.linspace(-15, 15, 200)
        # make a grid matrix of theta and thetadot
        theta, thetadot = np.meshgrid(ntheta, nthetadot)
        # combine the two matrix into a 2D array
        s = np.stack((theta, thetadot), axis=-1)
        s = s.reshape(-1, 2)

        V = agent.get_state_value(s)
        V_rs = V.reshape(200, 200)
        # flip x-axis of V_rs
        V_rs = np.flip(V_rs, axis=0)
        p.plot_table(V_rs, title="State-Value Function of DQN\n" + exp[4:], save=True, colorbar_label="value",
                     xlabels=np.linspace(-np.pi, np.pi, 5), ylabels=np.linspace(-15, 15, 5)[::-1],
                     state_names=['theta', 'thetadot'])

        policy_matrix = agent.get_action(s, 0)
        # action to tau
        policy_matrix = env.a_to_u(policy_matrix)
        policy_matrix = policy_matrix.reshape(200, 200)
        # flip x-axis of policy_matrix
        policy_matrix = np.flip(policy_matrix, axis=0)
        p.plot_table(policy_matrix, title="Policy of DQN\n" + exp[4:], save=True, colorbar_label="tau",
                     xlabels=np.linspace(-np.pi, np.pi, 5), ylabels=np.linspace(-15, 15, 5)[::-1],
                     state_names=['theta', 'thetadot'])

        #
        ######################################

        ######################################
        #
        #   EXAMPLE OF CREATING A PLOT
        #

        # # Initialize simulation
        s = env.reset()

        # Create dict to store data from simulation
        data = {
            't': [0],
            's': [s],
            'a': [],
            'r': [],
        }

        # Simulate until episode is done
        done = False
        while not done:
            a = policy(s)
            (s, r, done) = env.step(a)
            data['t'].append(data['t'][-1] + 1)
            data['s'].append(s)
            data['a'].append(a)
            data['r'].append(r)

        # Parse data from simulation
        data['s'] = np.array(data['s'])
        theta = data['s'][:, 0]
        thetadot = data['s'][:, 1]
        tau = [env._a_to_u(a) for a in data['a']]

        # Plot data and save to png file
        fig, ax = plt.subplots(3, 1, figsize=(10, 10))
        ax[0].plot(data['t'], theta, label='theta')
        ax[0].plot(data['t'], thetadot, label='thetadot')
        ax[0].legend()
        ax[1].plot(data['t'][:-1], tau, label='tau')
        ax[1].legend()
        ax[2].plot(data['t'][:-1], data['r'], label='r')
        ax[2].legend()
        ax[2].set_xlabel('time step')
        plt.tight_layout()
        plt.savefig('figures/test_discreteaction_pendulum_' + exp + '.png')
        plt.show()

    #
    ######################################


if __name__ == '__main__':
    main()
