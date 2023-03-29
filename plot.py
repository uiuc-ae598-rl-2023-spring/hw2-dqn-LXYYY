import os
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
from matplotlib.cm import get_cmap


class Plot:
    Plots = {}

    @staticmethod
    def save_all_plots(path):
        if not os.path.exists(path):
            os.makedirs(path)
        for scene in Plot.Plots:
            for algorithm in Plot.Plots[scene]:
                Plot.Plots[scene][algorithm].save_plot(path)

    def save_plot(self, path):
        # check if path exists
        path = os.path.join(path, self.plot_name)
        if not os.path.exists(path):
            os.makedirs(path)
        # save plot data
        for key in self.values:
            np.save(os.path.join(path, key), self.values[key])

    def __init__(self, env, scene, algorithm):
        self.values = {}
        self.plot_style = {}
        self.plot_args = {}
        self.experiment = algorithm
        self.scene = scene
        self.env = env
        self.plot_name = scene + '_' + algorithm
        if scene not in self.Plots:
            self.Plots[scene] = {}
        self.Plots[scene][algorithm] = self

    def make_dirs(self):
        path = os.path.join(os.getcwd(), 'figures',
                            self.scene, self.experiment)
        if not os.path.exists(path):
            os.makedirs(path)

    def add(self, key, value, plot=None, **kwargs):
        if key not in self.values:
            self.values[key] = []
            self.plot_style[key] = None
            self.plot_args[key] = {}
        self.values[key].append(value)
        if plot is not None:
            self.plot_style[key] = plot
        if kwargs:
            self.plot_args[key] = kwargs

    def _plot_grid(self):
        # set plot x limit from -1 to state shape 0
        plt.xlim(-1, self.env.state_shape[0])
        # set plot y limit from -1 to state shape 1
        plt.ylim(-1, self.env.state_shape[1])
        # generate x and y coordinates grid
        x = np.linspace(
            0, self.env.state_shape[0] - 1, self.env.state_shape[0])
        X, Y = np.meshgrid(x, x)
        # draw grid
        plt.scatter(X, Y, marker=MarkerStyle('s'), c='k')

    def plot(self, save=False, **kwargs):
        self.make_dirs()
        import matplotlib.pyplot as plt
        for k, v in self.values.items():
            args = self.plot_args[k]
            if 'xlabel' in args:
                plt.xlabel(args['xlabel'])
                del args['xlabel']
            if 'ylabel' in args:
                plt.ylabel(args['ylabel'])
                del args['ylabel']
            if 'title' in args:
                plt.title(args['title'])
                del args['title']
            if self.plot_style[k] is None:
                plt.plot(v, label=k, **kwargs, **self.plot_args[k])
            elif self.plot_style[k] == 'trajectory':
                if self.scene == 'gridworld':
                    self._plot_grid()
                    v_np = np.array(v)
                    v_np[:, 1] = 4 - v_np[:, 1]
                    # plot arrows
                    plt.quiver(v_np[:-1, 0], v_np[:-1, 1], v_np[1:, 0] - v_np[:-1, 0], v_np[1:, 1] - v_np[:-1, 1],
                               scale_units='xy',
                               angles='xy', scale=1, **kwargs, **self.plot_args[k])
                    plt.xlabel(self.env.state_names[0])
                    plt.ylabel(self.env.state_names[1])
                elif self.scene == 'pendulum':
                    theta = np.array(v)[:, 0]
                    x = np.sin(theta)
                    y = np.cos(theta)
                    # draw x and y separately
                    plt.plot(x, label='x', color='r', **
                    kwargs, **self.plot_args[k])
                    plt.plot(y, label='y', color='b', **
                    kwargs, **self.plot_args[k])
            plt.legend()
            if save:
                plt.savefig('figures/' + self.scene + '/' +
                            self.experiment + '/' + k + '.png')
            plt.show()

    def plot_policy(self, policy, save=False):
        if self.scene == 'gridworld':
            self._plot_grid()
            # arrows
            for i in range(len(policy) - 1):
                x, y = self.env.get_pos(i)
                # draw policy as arrows, 0: right, 1: up, 2: left, 3: down
                l = 0.3
                dxdy = [[l, 0], [0, l], [-l, 0], [0, -l]]
                plt.arrow(x, 4 - y, dxdy[policy[i]][0], dxdy[policy[i]][1], head_width=0.1, head_length=0.1, fc='k',
                          ec='k')
        elif self.scene == 'pendulum':
            # policy to color map
            cmap = get_cmap('jet')
            # get max and min q values
            max_p = np.max(policy)
            min_p = np.min(policy)
            # draw q values as colors
            for i in range(len(policy)):
                x, y = self.env.get_pos(i)
                # draw boxes and set size to a grid box without gap, and shape to square
                plt.scatter(x, y, marker=MarkerStyle('s'), c=cmap(
                    (policy[i] - min_p) / (max_p - min_p)), s=100)
            # set x and y labels
            plt.xlabel(r'$\theta$')
            plt.ylabel(r'$\dot{\theta}$')
            # show color bar and set max and min values to -env.max_tau and env.max_tau
            cbar = plt.colorbar()
            cbar.set_ticks([0, 1])
            # set color bar labels with precision 1
            cbar.set_ticklabels(
                ['%.1f' % -self.env.max_tau, '%.1f' % self.env.max_tau])
            # set color bar label
            cbar.set_label(r'$\tau$')
        plt.title('Policy of ' + self.experiment + ' in ' + self.scene)
        if save:
            plt.savefig('figures/' + self.scene + '/' +
                        self.experiment + '/policy.png')
        plt.show()

    def plot_table(self, V, title, state_names=None, save=False, caption=None, colorbar_label=None, xlabels=None,
                   ylabels=None, **kwargs):
        if xlabels is not None and type(xlabels[0]) is not str:
            xlabels = ['%.2f' % x for x in xlabels]
        if ylabels is not None and type(ylabels[0]) is not str:
            ylabels = ['%.2f' % y for y in ylabels]

        plt.figure()
        plt.imshow(V, **kwargs)
        # set plot xticks
        if xlabels is not None:
            dim_x = V.shape[0]
            xticks = np.linspace(0, dim_x - 1, len(xlabels))
            plt.xticks(xticks, xlabels)
        # set plot yticks
        if ylabels is not None:
            dim_y = V.shape[1]
            yticks = np.linspace(0, dim_y - 1, len(ylabels))
            plt.yticks(yticks, ylabels)

        plt.title(title)
        # draw colorbar and set max and min values
        v_max = np.max(V)
        v_min = np.min(V)
        cbar = plt.colorbar()
        cbar.set_ticks([v_min, v_max])
        # set color bar labels with precision 1
        cbar.set_ticklabels(['%.1f' % v_min, '%.1f' % v_max])
        # set color bar label
        cbar.set_label(colorbar_label) if colorbar_label is not None else None
        # set x and y labels
        if state_names is None:
            state_names = self.env.state_names
        plt.xlabel(state_names[0])
        plt.ylabel(state_names[1])
        plt.title(title)
        # add caption
        if caption is not None:
            plt.figtext(0.5, 0.01, caption, wrap=True,
                        horizontalalignment='center', fontsize=12)
        if save:
            # replace spaces in title with underscores
            file_name = title.replace(' ', '_')
            file_name += '_' + caption.replace(' ', '_') if caption is not None else ''
            # replace \n with _
            file_name = file_name.replace('\n', '_')
            plt.savefig('figures/' + file_name + '.png')
        plt.show()

    def plot_state_value_function(self, V, title, s=None, save=False, state_names=None, caption=None, vlim=None):
        if vlim is None:
            vlim = 0
        plt.figure()
        if state_names is None:
            state_names = self.env.state_names
        # colormap from q values to colors
        cmap = get_cmap('jet')
        # get max and min q values
        max_v_min = np.min(V)
        max_v_max = np.max(V)
        if self.scene == 'gridworld':
            self._plot_grid()

        # if V is a 1d array
        if s is None:
            # draw q values as colors
            for i in range(len(V)):
                x, y = self.env.get_pos(i)
                y = self.env.state_shape[0] - 1 - \
                    y if self.scene == 'gridworld' else y
                # draw boxes and set size to a grid box without gap, and shape to square
                plt.scatter(x, y, marker=MarkerStyle('s'), c=cmap(
                    (V[i] - max_v_min) / (max_v_max - max_v_min)), s=1000)

        # if V is a 2d matrix
        elif s is not None:
            # increase min_v by vlim percent
            max_v_min += vlim * (max_v_max - max_v_min)
            d = (V - max_v_min) / (max_v_max - max_v_min)
            d[d < 0] = 0
            # draw boxes and set size to a grid box without gap, and shape to square
            plt.scatter(s[:, 0], s[:, 1], marker=MarkerStyle('s'), c=cmap(d), s=1000)

        # draw colorbar and set max and min values
        cbar = plt.colorbar()
        cbar.set_ticks([0, 1])
        # set color bar labels with precision 1
        cbar.set_ticklabels(['%.1f' % max_v_min, '%.1f' % max_v_max])
        # set color bar label
        cbar.set_label('value')
        # set x and y labels
        plt.xlabel(state_names[0])
        plt.ylabel(state_names[1])
        plt.title(title)
        # add caption
        if caption is not None:
            plt.figtext(0.5, 0.01, caption, wrap=True,
                        horizontalalignment='center', fontsize=12)
        if save:
            # replace spaces in title with underscores
            file_name = title.replace(' ', '_')
            file_name += '_' + caption.replace(' ', '_') if caption is not None else ''
            # replace \n with _
            file_name = file_name.replace('\n', '_')
            plt.savefig('figures/' + file_name + '.png')
        plt.show()

    def clear(self, k):
        # remove the key k from all dictionaries
        for i in range(len(k)):
            if k[i] in self.values:
                del self.values[k[i]]

    @staticmethod
    def plot_compare(scenes, algorithms, key, title, save=False, **kwargs):
        # get plot_interval and remove it from kwargs
        plot_interval = kwargs.get('plot_interval', False)
        del kwargs['plot_interval']
        import matplotlib.pyplot as plt
        plt.figure()

        for scene, algorithm in zip(scenes, algorithms):
            v = Plot.Plots[scene][algorithm].values[key]
            if plot_interval:
                v = np.array(v)
                plt.fill_between(range(len(v)), v - v.std(), v + v.std(), alpha=0.5, label=scene + '_' + algorithm,
                                 **kwargs)
            else:
                plt.plot(v, label=algorithm, **kwargs)

        plt.legend()
        plt.title(title)
        if save:
            plt.savefig('figures/' + title + '.png')
        plt.show()

    @staticmethod
    def get_all_plot_names():
        plots = []
        for scene in Plot.Plots:
            for algorithm in Plot.Plots[scene]:
                plots.append(scene + '_' + algorithm)
        return plots


def plot_learning_curves(rewards, name, save=False):
    plt.figure()
    for i in range(len(rewards)):
        plt.plot(rewards[i], label=name[i])
    plt.legend()
    plt.title('Learning Curves')
    if save:
        plt.savefig('figures/learning_curves.png')
    plt.show()
