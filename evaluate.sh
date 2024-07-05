#!/bin/bash
export OMP_NUM_THREADS=3  # speeds up MinkowskiEngine

# TEST
python main_instance_segmentation.py \
general.OW_task="$2" \
general.split="$1" \
general.experiment_name="OpenDistill3D_A" \
general.project_name="OpenDistill3D" \
general.checkpoint="$3" \
data/datasets=scannet200 \
general.eval_on_segments=true \
general.train_on_segments=true \
general.train_mode=false \
general.export=false