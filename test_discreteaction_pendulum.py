import random
import numpy as np
import matplotlib.pyplot as plt
import discreteaction_pendulum
import sys
import argparse
import torch.optim as optim

import dqn
import plot

load_ckp = True
ckp = 'dqn.pth'


# Main function with arguments
def main():

    # arg parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', action='store_true', help='load checkpoint')
    parser.add_argument('--save', action='store_true', help='save checkpoint')
    parser.add_argument('--exp', type=str, help='experiment name')

    exp = parser.parse_args().exp if parser.parse_args().exp else 'dqn'
    load_ckp = parser.parse_args().load if parser.parse_args().load else False
    ckp = exp+'.pth'

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

    hidden_size = 64
    gamma = 0.95
    lr = 0.001

    model = dqn.DQN(env.num_states, hidden_dim=hidden_size,
                    output_dim=env.num_actions)
    buffer = dqn.ReplayBuffer(
        buffer_size=10000, state_dim=env.num_states, batch_size=32)
    opt = optim.Adam(model.parameters(), lr=lr)

    agent = dqn.Agent(model, buffer, opt, env.num_states,
                      env.num_states, gamma)

    scores = None
    if load_ckp:
        agent.load_model(ckp)
    else:
        # Train agent
        scores = dqn.train(env, agent, num_episodes=1000, batch_size=32,
                           epsilon=1.0, epsilon_decay=0.99, epsilon_min=0.1, render=False)

        # save model
        agent.save_model(ckp)

    ######################################
    #
    #   EXAMPLE OF CREATING A VIDEO
    #

    policy = agent.get_policy()

    # Simulate an episode and save the result as an animated gif
    env.video(policy=policy,
              filename='figures/test_discreteaction_pendulum.gif')

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
        np.save('scores.npy', scores)

        plt.figure()
        plt.ylim(0, 100)
        plot.plot_learning_curves(
            scores, 'dqn_bs32_lr0.001_h64_g0.95', save=True)

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
    plt.savefig('figures/test_discreteaction_pendulum.png')
    plt.show()

    #
    ######################################


if __name__ == '__main__':
    main()
