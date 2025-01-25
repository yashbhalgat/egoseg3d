import argparse, os, json
from collections import defaultdict

import utils
from global_vars import SEG_VALID_COUNTS, IGNORE_CAT


def count_IDs(preds, area=True):
    class_names = preds["category_name2id"].keys()
    class_names = sorted(set(class_names).difference(IGNORE_CAT))
    class_ids = [preds["category_name2id"][x] for x in class_names]
    id_to_name = {v: k for k, v in preds["category_name2id"].items()}
    class_to_inst = defaultdict(set)
    for anno in preds["annotations"]:
        for seginfo in anno["segments_info"]:
            if area and seginfo["area"] < SEG_VALID_COUNTS:
                continue
            if seginfo["category_id"] in class_ids:
                # class_to_inst[seginfo["category_id"]].add(seginfo["id"])
                class_to_inst[id_to_name[seginfo["category_id"]]].add(seginfo["id"])
    ID_counts = {k: len(v) for k, v in class_to_inst.items()}
    return ID_counts


if __name__ == '__main__':
    vids = list(set(['_'.join(x.split('_')[:2]) for x in os.listdir('../scripts/out/visor/') if '.json' in x]))
    vids = [x for x in vids if x not in ['P14_05', 'P03_13', 'P30_112']]

    deva_ID_counts_all, ours_ID_counts_all = {}, {}
    for vid in vids:
        pid = vid.split('_')[0]

        deva_json = f'./out/{pid}/{vid}/segmaps/deva_OWLv2_s5/pred.json'
        ours_json = f'./out/{pid}/{vid}/segmaps/tracked-final-bv2-bs10/pred.json'
        deva_preds = utils.read_json(deva_json)
        ours_preds = utils.read_json(ours_json)

        deva_ID_counts = count_IDs(deva_preds)
        ours_ID_counts = count_IDs(ours_preds, area=False)
        deva_ID_counts_all[vid] = deva_ID_counts
        ours_ID_counts_all[vid] = ours_ID_counts

    # get class-wise average over all videos
    deva_ID_counts_avg = defaultdict(float)
    ours_ID_counts_avg = defaultdict(float)
    class_video_count = defaultdict(int)
    for vid in vids:
        for k, v in deva_ID_counts_all[vid].items():
            deva_ID_counts_avg[k] += v
        for k, v in ours_ID_counts_all[vid].items():
            ours_ID_counts_avg[k] += v
        for k in deva_ID_counts_all[vid].keys():
            class_video_count[k] += 1
        
    for k in deva_ID_counts_avg.keys():
        deva_ID_counts_avg[k] /= class_video_count[k]
    for k in ours_ID_counts_avg.keys():
        ours_ID_counts_avg[k] /= class_video_count[k]

    # sort class names by deva counts
    # sorted_cls = sorted(deva_ID_counts_avg.keys(), key=lambda x: deva_ID_counts_avg[x], reverse=True)
    sorted_cls = sorted(deva_ID_counts_avg.keys(), key=lambda x: class_video_count[x], reverse=True)

    # print class-wise counts
    print('Class: Deva count | Ours count | Video count')
    for k in sorted_cls:
        print(f'{k} & {deva_ID_counts_avg[k]:.2f} & {ours_ID_counts_avg[k]:.2f} & {class_video_count[k]}')
    print('done')

    # print for every video
    for k in sorted_cls:
        print(f'{k}:')
        for vid in vids:
            if k not in deva_ID_counts_all[vid] or k not in ours_ID_counts_all[vid]:
                continue
            print(f'{vid}: {deva_ID_counts_all[vid][k]} | {ours_ID_counts_all[vid][k]}')
        print()