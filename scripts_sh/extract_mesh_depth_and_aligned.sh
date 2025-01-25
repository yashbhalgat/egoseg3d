# Usage: bash ./scripts_sh/extract_mesh_depth_and_aligned.sh <vid>

ROOT="./scripts"
cd $ROOT

VID=$1

python extract_mesh_depth.py --vid=$VID
python extract_aligned_depth.py --vid=$VID

