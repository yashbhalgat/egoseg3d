import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import numpy as np

def parallelCoordinatesPlot(title, N, data, category, ynames, colors=None, category_names=None):
    """
    A legend is added, if category_names is not None.

    :param title: The title of the plot.
    :param N: Number of data sets (i.e., lines).
    :param data: A list containing one array per parallel axis, each containing N data points.
    :param category: An array containing the category of each data set.
    :param category_names: Labels of the categories. Must have the same length as set(category).
    :param ynames: The labels of the parallel axes.
    :param colors: A colormap to use.
    :return:
    """

    fig, host = plt.subplots()

    # organize the data
    ys = np.dstack(data)[0]
    ymins = ys.min(axis=0)
    ymaxs = ys.max(axis=0)
    dys = ymaxs - ymins
    ymins -= dys * 0.015  # add 1% padding below and above
    ymaxs += dys * 0.015
    dys = ymaxs - ymins

    # transform all data to be compatible with the main axis
    zs = np.zeros_like(ys)
    zs[:, 0] = ys[:, 0]
    zs[:, 1:] = (ys[:, 1:] - ymins[1:]) / dys[1:] * dys[0] + ymins[0]
    zs -= np.random.uniform(0, 1, zs.shape)  # to avoid overplotting

    axes = [host] + [host.twinx() for i in range(ys.shape[1] - 1)]
    for i, ax in enumerate(axes):
        ax.set_ylim(ymins[i], ymaxs[i])
        if i < len(axes) - 1:
            # change the first ytick label only from 0 to 1. i.e., 0 -> 1
            # first get the yticks
            # yticks = ax.get_yticks()
            # # find the index of 0 in yticks
            # idx = np.where(yticks == 0)[0][0]
            # yticks[idx] = data[i].min()
            # ax.set_yticks(yticks)
            # # remove the largest ytick label
            # yticks = ax.get_yticks()
            # ax.set_yticks(yticks[1:-1])
            # ax.set_yticklabels([f'{int(tick)}' for tick in yticks[1:-1]])
            ### ONLY show yticks for which we have data
            yticks = ax.get_yticks()
            uniq_data = np.unique(data[i])
            yticks = uniq_data.tolist()
            ax.set_yticks(yticks)
            ax.set_yticklabels([f'{int(tick)}' for tick in yticks])

        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        if ax != host:
            ax.spines['left'].set_visible(False)
            ax.yaxis.set_ticks_position('right')
            ax.spines["right"].set_position(("axes", i / (ys.shape[1] - 1)))

        # set ytick font size for each axis
        ax.tick_params(axis='y', which='both', labelsize=25)



    host.set_xlim(0, ys.shape[1] - 1)
    host.set_xticks(range(ys.shape[1]))
    host.set_xticklabels(ynames, fontsize=30)
    host.tick_params(axis='x', which='major', pad=7)
    host.spines['right'].set_visible(False)
    host.xaxis.tick_top()

    if colors is None:
        colors = plt.cm.tab10.colors
    if category_names is not None:
        legend_handles = [None for _ in category_names]
    else:
        legend_handles = [None for _ in set(category)]
    for j in range(N):
        verts = list(zip([x for x in np.linspace(0, len(ys) - 1, len(ys) * 3 - 2, endpoint=True)],
                         np.repeat(zs[j, :], 3)[1:-1]))
        codes = [Path.MOVETO] + [Path.CURVE4 for _ in range(len(verts) - 1)]
        path = Path(verts, codes)
        patch = patches.PathPatch(path, facecolor='none', lw=1, edgecolor=colors[category[j] - 1])
        legend_handles[category[j] - 1] = patch
        host.add_patch(patch)

    if category_names is not None:
        host.legend(legend_handles, category_names,
                    loc='lower center', bbox_to_anchor=(0.5, -0.18),
                    ncol=len(category_names), fancybox=True, shadow=True)

    plt.tight_layout()
    plt.show()


def parallelCoordinatesPlot_log(title, N, data, category, ynames, colors=None, category_names=None):
    """
    A legend is added, if category_names is not None.

    :param title: The title of the plot.
    :param N: Number of data sets (i.e., lines).
    :param data: A list containing one array per parallel axis, each containing N data points.
    :param category: An array containing the category of each data set.
    :param category_names: Labels of the categories. Must have the same length as set(category).
    :param ynames: The labels of the parallel axes.
    :param colors: A colormap to use.
    :return:
    """

    fig, host = plt.subplots()

    # Ensure data has positive values before log transformation
    data = [np.where(d <= 0, np.min(d[d > 0]) * 0.1, d) for d in data]

    # Apply log scale to all data except the last one
    log_data = [np.log10(d) if i < len(data) - 1 else d for i, d in enumerate(data)]

    # Organize the data
    ys = np.dstack(log_data)[0]
    ymins = ys.min(axis=0)
    ymaxs = ys.max(axis=0)
    dys = ymaxs - ymins
    ymins -= dys * 0.05  # add 5% padding below and above
    ymaxs += dys * 0.05
    dys = ymaxs - ymins

    # Transform all data to be compatible with the main axis
    zs = np.zeros_like(ys)
    zs[:, 0] = ys[:, 0]
    zs[:, 1:] = (ys[:, 1:] - ymins[1:]) / dys[1:] * dys[0] + ymins[0]
    # zs += np.random.uniform(-1, 1, zs.shape)  # to avoid overplotting

    axes = [host] + [host.twinx() for i in range(ys.shape[1] - 1)]
    for i, ax in enumerate(axes):
        if i < len(axes) - 1:
            ax.set_yscale('log')
            ymin, ymax = 10**ymins[i], 10**ymaxs[i]
            ax.set_ylim(ymin, ymax)
            ticks = [10**j for j in range(int(np.floor(np.log10(ymin))), int(np.ceil(np.log10(ymax))) + 1)]
            ax.set_yticks(ticks)
            ax.set_yticklabels([f'{int(tick)}' for tick in ticks])
        else:
            ymin, ymax = ymins[i], ymaxs[i]
            ax.set_ylim(ymin, ymax)

        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        if ax != host:
            ax.spines['left'].set_visible(False)
            ax.yaxis.set_ticks_position('right')
            ax.spines["right"].set_position(("axes", i / (ys.shape[1] - 1)))

    host.set_xlim(0, ys.shape[1] - 1)
    host.set_xticks(range(ys.shape[1]))
    host.set_xticklabels(ynames, fontsize=14)
    host.tick_params(axis='x', which='major', pad=7)
    host.spines['right'].set_visible(False)
    host.xaxis.tick_top()
    host.set_title(title, fontsize=18)

    if colors is None:
        colors = plt.cm.tab10.colors
    if category_names is not None:
        legend_handles = [None for _ in category_names]
    else:
        legend_handles = [None for _ in set(category)]
    for j in range(N):
        verts = list(zip([x for x in np.linspace(0, len(ys) - 1, len(ys) * 3 - 2, endpoint=True)],
                         np.repeat(zs[j, :], 3)[1:-1]))
        codes = [Path.MOVETO] + [Path.CURVE4 for _ in range(len(verts) - 1)]
        path = Path(verts, codes)
        patch = patches.PathPatch(path, facecolor='none', lw=1, edgecolor=colors[category[j] - 1])
        legend_handles[category[j] - 1] = patch
        host.add_patch(patch)

    if category_names is not None:
        host.legend(legend_handles, category_names,
                    loc='lower center', bbox_to_anchor=(0.5, -0.18),
                    ncol=len(category_names), fancybox=True, shadow=True)

    plt.tight_layout()
    plt.show()


