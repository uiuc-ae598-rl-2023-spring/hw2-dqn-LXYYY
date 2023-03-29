from matplotlib import pyplot as plt
import numpy as np
import os

from utils import *

plt.ion()

gamma = 0.95
user_input_ablation = False


def get_exp_and_run_from_string(s):
    s = s[11:-4]
    return s.split('_r:')[0], int(s.split('_r:')[1]) if '_r:' in s else 0


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
        abl_exp, _ = get_exp_and_run_from_string(f)
        if abl_exp not in exps:
            exps.append(abl_exp)

        path = os.path.join(data_folder, f)
        scores = np.load(path)
        discount_factors = np.array([gamma ** i for i in range(scores.shape[1])])
        rewards = np.zeros(scores.shape[0])
        und_rewards = np.zeros(scores.shape[0])
        for i in range(scores.shape[0]):
            rewards[i] = np.sum(scores[i, :] * discount_factors)
            und_rewards[i] = np.sum(scores[i, :])

        if abl_exp not in rewards_exp:
            rewards_exp[abl_exp] = []
            und_rewards_exp[abl_exp] = []

        rewards_exp[abl_exp].append(rewards)
        und_rewards_exp[abl_exp].append(und_rewards)

    for exp in exps:
        print('Plotting learning curve of experiment: ' + exp)
        # plot rewards and und_rewards in two subplots
        plt.figure()
        plt.subplot(2, 1, 1)
        plot_learning_curve(rewards_exp[exp], 'gamma=' + str(gamma))
        plt.title('Learning curve of \n' + exp)
        plt.ylim(-10, 40)
        # plt.xlabel('episode')
        plt.ylabel('reward')
        plt.legend()
        plt.subplot(2, 1, 2)
        plot_learning_curve(und_rewards_exp[exp], 'undiscounted')
        plt.ylim(-10, 80)
        plt.xlabel('episodes')
        plt.ylabel('reward')
        plt.legend()
        plt.savefig('figures/learning_curve_' + exp + '.png')
        plt.show()
        plt.close()

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
            for abl_exp in a:
                abl_exps.append(abl_exp[4:])
            abl_exps_all.append(abl_exps)

    for i, abl_exps in enumerate(abl_exps_all):
        print('Plotting learning curve of ablation experiments {}: '.format(i))
        for abl_exp in abl_exps:
            print(abl_exp)

        # plot learning curves
        plt.figure()
        for abl_exp in abl_exps:
            if abl_exp not in exps:
                raise ValueError('Experiment ' + abl_exp + ' not found!')
            # convert exp to more readable format
            label = abl_exp.replace('lr:0.001', ' ')
            label = label.replace('h:', '')
            label = label.replace('tu:1_', 'wo. target Q, ')
            label = label.replace('tu:100_', 'w. target Q, ')
            label = label.replace('bs:32', '')
            label = label.replace('rs:32', 'wo. replay')
            label = label.replace('rs:10000', 'w. replay')
            label = label.replace('e:1_ed:0.99_em:0.3', 'w. epsilon decay, ')
            label = label.replace('e:0.3_ed:1_em:0', 'wo. epsilon decay, ')

            label = label.replace('_', '')

            plot_learning_curve(rewards_exp[abl_exp], label)

        # set all curves alpha to 0.5
        for line in plt.gca().get_lines():
            line.set_alpha(0.5)


        plt.title('Learning curve of ablation experiments')
        plt.ylim(-10, 10)
        plt.xlabel('episode')
        plt.ylabel('reward')
        plt.legend()
        plt.savefig('figures/learning_curve_ablation_{}.png'.format(i))
        plt.show()


def compute_mean_and_std(rewards):
    # is rewards is a list, convert it to numpy array
    if isinstance(rewards, list):
        rewards = np.array(rewards)
    # compute mean and std
    mean = np.mean(rewards, axis=0)
    std = np.std(rewards, axis=0)
    return mean, std


def plot_mean_and_std(mean, std, label, no_std=False):
    # plot mean and std
    plt.plot(mean, label=label)
    if not no_std:
        plt.fill_between(np.arange(mean.shape[0]), mean - std, mean + std, alpha=0.2)


def plot_learning_curve(rewards, label, no_std=False):
    # plot learning curve
    mean, std = compute_mean_and_std(rewards)
    plot_mean_and_std(mean, std, label, no_std=no_std)
    plt.xlim(0, len(mean))


if __name__ == '__main__':
    plot_ablation()
