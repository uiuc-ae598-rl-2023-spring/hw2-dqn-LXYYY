from matplotlib import pyplot as plt
import numpy as np
import os

from utils import *

plt.ion()

gamma = 0.95
user_input_ablation = False


def get_setting_from_string(s):
    return s[11:-4]


# main function
def plot_ablation():
    # scan the data folder
    data_folder = 'data'
    files = os.listdir(data_folder)
    # files start with scores_DQN and end with .npy
    files = [f for f in files if f.startswith('scores_DQN') and f.endswith('.npy')]

    rewards_exp = {}
    und_rewards_exp = {}
    exps = []
    for f in files:
        # get all string after scores_DQN_
        exp = get_setting_from_string(f)
        print('Plotting learning curve of ' + exp + '...')
        exps.append(exp)

        path = os.path.join(data_folder, f)
        scores = np.load(path)
        discount_factors = np.array([gamma ** i for i in range(scores.shape[1])])
        rewards = np.zeros(scores.shape[0])
        und_rewards = np.zeros(scores.shape[0])
        for i in range(scores.shape[0]):
            rewards[i] = np.sum(scores[i, :] * discount_factors)
            und_rewards[i] = np.sum(scores[i, :])

        rewards_exp[exp] = rewards
        und_rewards_exp[exp] = und_rewards

        # plot rewards and und_rewards in two subplots
        plt.figure()
        plt.subplot(2, 1, 1)
        plot_learning_curve(rewards_exp[exp], 20, 'gamma=' + str(gamma))
        plt.title('Learning curve of \n' + exp)
        plt.ylim(-10, 40)
        # plt.xlabel('episode')
        plt.ylabel('reward')
        plt.legend()
        plt.subplot(2, 1, 2)
        plot_learning_curve(und_rewards_exp[exp], 20, 'undiscounted')
        plt.ylim(-10, 80)
        plt.xlabel('episodes')
        plt.ylabel('reward')
        plt.legend()
        plt.savefig('figures/learning_curve_' + exp + '.png')
        plt.show()

    abl_exps = []
    abl_exps_all = []
    if user_input_ablation:
        # plot learning curves of selected ablation experiments
        # ask for user input to select the experiments
        print('Please select the experiments to plot ablation learning curves:')
        # print all experiments
        for i in range(len(exps)):
            print(str(i) + ': ' + exps[i])
        # get user input
        abl = input('Please input the indices of experiments separated by space:')
        abl = abl.split(' ')
        abl_exps = [exps[int(i)] for i in abl]
        abl_exps_all = [abl_exps]
    else:
        # get learning_curve_comp from the file
        abls = read_json_file('ablation.json')['learning_curve_comp']
        for a in abls:
            abl_exps = []
            for exp in a:
                abl_exps.append(exp[4:])
            abl_exps_all.append(abl_exps)

    for i, abl_exps in enumerate(abl_exps_all):
        print('Plotting learning curve of ablation experiments:')
        for exp in abl_exps:
            print(exp)

        # plot learning curves
        plt.figure()
        for exp in abl_exps:
            # convert exp to more readable format
            label = exp.replace('lr:0.001', ' ')
            label = label.replace('h:', '')
            label = label.replace('tu:1_', 'wo. target Q, ')
            label = label.replace('tu:100_', 'w. target Q, ')
            label = label.replace('bs:32', '')
            label = label.replace('rs:32', 'wo. replay')
            label = label.replace('rs:10000', 'w. replay')
            label = label.replace('e:1_ed:0.99_em:0.3', 'w. epsilon decay ')
            label = label.replace('e:0.3_ed:1_em:0', 'wo. epsilon decay ')

            label = label.replace('_', '')

            plot_learning_curve(rewards_exp[exp], 20, label, no_std=True)
        plt.title('Learning curve of ablation experiments')
        plt.ylim(-10, 10)
        plt.xlabel('episode')
        plt.ylabel('reward')
        plt.legend()
        plt.savefig('figures/learning_curve_ablation_{}.png'.format(i))
        plt.show()


def compute_mean_and_std(rewards, n):
    # compute mean and standard deviation of rewards
    # n is the number of episodes to compute mean and std
    mean = np.zeros(rewards.shape[0] // n)
    std = np.zeros(rewards.shape[0] // n)
    for i in range(mean.shape[0]):
        mean[i] = np.mean(rewards[i * n:(i + 1) * n])
        std[i] = np.std(rewards[i * n:(i + 1) * n])
    return mean, std


def plot_mean_and_std(mean, std, label, no_std=False):
    # plot mean and std
    plt.plot(mean, label=label)
    if not no_std:
        plt.fill_between(np.arange(mean.shape[0]), mean - std, mean + std, alpha=0.2)


def plot_learning_curve(rewards, n, label, no_std=False):
    # plot learning curve
    mean, std = compute_mean_and_std(rewards, n)
    plot_mean_and_std(mean, std, label, no_std=no_std)
    # multiple ticks by n
    plt.xticks(np.arange(0, mean.shape[0] + 1, 10), np.arange(0, (mean.shape[0] + 1) * n, 10 * n))
    # plt.xlim(-n, mean.shape[0] + n)


if __name__ == '__main__':
    plot_ablation()
