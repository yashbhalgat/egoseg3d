import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os, json
from glob import glob

from utils import read_json

def path2vid(p):
    start = p.find('out/P') + 4
    pid, vid = p[start:].split('/')[:2]
    return vid

def plot(deva_array, ours_array, array_name):
    sns.set_theme(style="whitegrid")
    # use seaborn for pretty plot. Both arrays in the same plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 5), sharey=True)
    sns.lineplot(x=np.arange(0.05, 1, 0.05), y=deva_array, ax=ax, marker="o", markersize=12, linewidth=2.5, label='DEVA')
    sns.lineplot(x=np.arange(0.05, 1, 0.05), y=ours_array, ax=ax, marker="o", markersize=12, linewidth=2.5, label='Ours')
    ax.set_title(array_name, fontsize=25)
    ax.set_xlabel("IoU threshold", fontsize=16)
    ax.set_ylabel("Score", fontsize=16)
    ax.set_xticks(np.arange(0.05, 1, 0.05))
    plt.legend(fontsize=16)
    plt.tight_layout()
    plt.savefig(f'plot_results/{array_name}.pdf')

def plot_together(deva_overall_arrays, ours_overall_arrays):
    # all 3 plots should be in the same figure. And overall aspect ratio should be horizontal
    # every axis should have its own y-axis
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    for i, k in enumerate(['HOTA', 'AssA']):
        sns.lineplot(x=np.arange(0.05, 1, 0.05), y=deva_overall_arrays[k], ax=ax[i], marker="o", markersize=8, linewidth=2.5, label='DEVA')
        sns.lineplot(x=np.arange(0.05, 1, 0.05), y=ours_overall_arrays[k], ax=ax[i], marker="o", markersize=8, linewidth=2.5, label='Ours')
        ax[i].set_title(k, fontsize=25)
        ax[i].set_xlabel("IoU threshold", fontsize=16)
        ax[i].set_ylabel("Score", fontsize=16)
        # ax[i].set_xticks(np.arange(0.05, 1, 0.05))
        # show only every 2nd tick
        ax[i].set_xticks(np.arange(0.05, 1, 0.1))
        min_val, max_val = min(min(deva_overall_arrays[k]), min(ours_overall_arrays[k])), max(max(deva_overall_arrays[k]), max(ours_overall_arrays[k]))
        if k == 'HOTA':
            ax[i].set_ylim(min_val - 0.01, max_val + 0.01)
        else:
            ax[i].set_ylim(min_val - 0.03, max_val + 0.03)
        ax[i].legend(fontsize=16)
    plt.tight_layout()
    plt.savefig(f'plot_results/all.pdf')


if __name__ == "__main__":
    deva = {path2vid(x): x for x in glob('../scripts/out/*/*/segmaps/deva_OWLv2_s5/hota-shortlist-VisorTrue.json')}
    ours = {path2vid(x): x for x in glob('../scripts/out/*/*/segmaps/tracked-final-bv2-bs10/hota.json')}

    vids = set(['_'.join(x.split('_')[:2]) for x in os.listdir('../scripts/out/visor/') if '.json' in x])

    deva_overall_arrays = {'HOTA': 0, 'DetA': 0, 'AssA': 0}
    ours_overall_arrays = {'HOTA': 0, 'DetA': 0, 'AssA': 0}
    count = 0
    for vid in vids:
        if vid == 'P14_05' or vid == 'P03_13' or vid == 'P30_112':
            continue
        deva_vid = read_json(deva[vid])
        ours_vid = read_json(ours[vid])
        for k in ['HOTA', 'DetA', 'AssA']:
            deva_overall_arrays[k] += np.array(deva_vid[k])
            ours_overall_arrays[k] += np.array(ours_vid[k])
        count += 1

    for k in deva_overall_arrays.keys():
        deva_overall_arrays[k] /= count
        ours_overall_arrays[k] /= count

    plot_together(deva_overall_arrays, ours_overall_arrays)
    # for k in deva_overall_arrays.keys():
    #     plot(deva_overall_arrays[k], ours_overall_arrays[k], k)
