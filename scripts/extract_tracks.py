from tracking import *
import tracking, utils, scenevis
from utils import tqdm, create_colormap_from_file
from pathlib import Path
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vid", type=str)
    parser.add_argument(
        "--root", type=str, default="/work/vadim/workspace/experiments/OSNOM-Lang/out"
    )
    parser.add_argument("--no_scaling", action="store_true")
    parser.add_argument("--deva_segment", type=str, default="deva_OWLv2_s5")
    parser.add_argument(
        "--fts_type", type=str, default="dino_s5"  # dino_s5, masa_dino_s5
    )
    parser.add_argument("--exp", type=str, default="")
    parser.add_argument("--alpha", type=float, default=30)
    parser.add_argument("--beta_c", type=float, default=10000)
    parser.add_argument("--beta_v", type=float, default=2)
    parser.add_argument("--beta_l", type=float, default=10)
    parser.add_argument("--beta_s", type=float, default=10)
    parser.add_argument("-f", type=str)

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()
    root = Path(args.root)
    vid = args.vid
    pid = vid.split("_")[0]
    # pid = "." # for Ego4D, no pid

    if not args.no_scaling:
        scaling_factor = utils.read_json(root / Path("meta.json"))["rescale_scores"][
            vid
        ]
    else:
        scaling_factor = 1

    print(f"Scaling factor: {scaling_factor}")
    print(
        f"alpha {args.alpha}. beta_c: {args.beta_c}. beta_v: {args.beta_v}. beta_l: {args.beta_l}. beta_s: {args.beta_s}"
    )

    if args.exp != "":
        exp = args.exp
    else:
        exp = "tracked"

    dir_out = root / Path(f"{pid}/{vid}/segmaps/{exp}")
    os.makedirs(dir_out, exist_ok=True)
    utils.save_as_json(dir_out / "args.json", vars(args))

    cmap = create_colormap_from_file()

    # the default deva loader references "s5"
    # if changed here, then must be changed also when initialising "Observations"
    # with "dir_fts" pointing to directory consistent with deva loader
    # i.e. the features are extracted based on DEVA segmentations
    deva_loader = DevaLoader(root=f"./out/{pid}/{vid}/segmaps/{args.deva_segment}/")

    # initialise observations per frame
    scene = Scene(vid)
    # valid_frames = set(deva_loader.frame2id).intersection(scene.frames)
    valid_frames = sorted(deva_loader.frame2id)[:]

    # adjust frame range for debugging, right now extracts all frames
    frames_selected = sorted(valid_frames)[0::1]

    observations = Observations(
        scene,
        deva_loader,
        scaling_factor=scaling_factor,
        dir_fts=root / Path(f"{pid}/{vid}/features/{args.fts_type}"),
    )

    print("Adding observations")
    for frame in tqdm(sorted(frames_selected)):
        observations.add_by_frame(frame)

    # initialise tracker, add observations
    tracker = Tracker(
        alpha=args.alpha,
        beta_v=args.beta_v,
        beta_l=args.beta_l,
        beta_c=args.beta_c,
        beta_s=args.beta_s,
    )
    print("Tracking...")
    for frame in tqdm(sorted(observations.frames)):
        tracker.add_observations(observations[frame])

    # export tracker results
    instances3d, inst2cat, locations3d = tracker.calc_instances(deva_loader)

    tracker.export(
        dir_out,
        instances3d,
        inst2cat,
        locations3d,
        deva_loader.catname2id,
        valid_frames,
    )
