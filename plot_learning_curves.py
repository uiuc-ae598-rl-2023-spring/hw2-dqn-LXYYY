from matplotlib import pyplot as plt
import numpy as np
import os

gamma = 0.95


# main function
def main():
    # scan the data folder
    data_folder = 'data'
    files = os.listdir(data_folder)
    # files start with scores_DQN and end with .npy
    files = [f for f in files if f.startswith('scores_DQN') and f.endswith('.npy')]

    for f in files:
        # get all string after scores_DQN_
        exp = f[11:-4]

        path = os.path.join(data_folder, f)
        scores = np.load(path)
        discount_factors = np.array([gamma ** i for i in range(scores.shape[1])])
        rewards = np.zeros(scores.shape[0])
        for i in range(scores.shape[0]):
            rewards[i] = np.sum(scores[i, :] * discount_factors)

        # plot learning curves
        plt.figure()
        plt.plot(rewards)
        plt.title('Learning curve of ' + exp)
        plt.ylim(0, 100)
        plt.xlabel('episode')
        plt.ylabel('reward')
        plt.savefig('figures/learning_curve_' + exp + '.png')
        plt.legend('gamma=' + str(gamma))
        plt.show()


if __name__ == '__main__':
    main()
