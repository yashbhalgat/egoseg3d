
ROOT="./out/mesh"

VIDEO_IDS=$1

# conda activate colmap

for VID in $VIDEO_IDS; do
    echo $VID

    DATASET_PATH=$ROOT/$VID
    mkdir $DATASET_PATH/dense

    colmap image_undistorter \
        --image_path $DATASET_PATH/images \
        --input_path $DATASET_PATH/sparse/0 \
        --output_path $DATASET_PATH/dense \
        --output_type COLMAP \
        --max_image_size 2000

    colmap patch_match_stereo \
        --workspace_path $DATASET_PATH/dense \
        --workspace_format COLMAP \
        --PatchMatchStereo.geom_consistency true

    colmap stereo_fusion \
        --workspace_path $DATASET_PATH/dense \
        --workspace_format COLMAP \
        --input_type geometric \
        --output_path $DATASET_PATH/dense/fused.ply

    colmap poisson_mesher \
        --input_path $DATASET_PATH/dense/fused.ply \
        --output_path $DATASET_PATH/dense/mesh.ply

done

