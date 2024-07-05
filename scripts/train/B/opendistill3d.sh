# !/bin/bash
export OMP_NUM_THREADS=3  # speeds up MinkowskiEngine

# TRAIN Task1
python main_instance_segmentation.py \
general.OW_task="task1" \
general.split="B" \
general.experiment_name="OpenDistill3D_B" \
general.project_name="OpenDitliss3D_train" \
general.use_conf_th=True \
general.margin=1.0 \
general.max_lr=0.0001 \
scheduler=onecyclelr_1stage \
general.use_mask_extractor=True \
general.use_scores_AL=True \
general.use_weight_func=True \
general.mask_conf_th=0.5


# TRAIN Task2
if [ ! -d "saved/OpenDistill3D_B_t2" ]; then
  echo "Copying saved/OpenDistill3D_B to saved/OpenDistill3D_B_t2..."
  cp -r saved/OpenDistill3D_B saved/OpenDistill3D_B_t2
else
  echo "saved/OpenDistill3D_B_t2 Already exists"
fi



python main_instance_segmentation.py \
general.OW_task="task2" \
general.split="B" \
general.experiment_name="OpenDistill3D_B_t2" \
general.project_name="OpenDitliss3D_train" \
general.use_conf_th=True \
general.margin=1.0 \
general.max_lr=0.0001 \
scheduler=onecyclelr_1stage \
general.use_mask_extractor=True \
general.use_scores_AL=True \
general.use_weight_func=True \
general.mask_conf_th=0.5 \
general.max_epochs=401

# TRAIN Task3
if [ ! -d "saved/OpenDistill3D_B_t3" ]; then
  echo "Copying saved/OpenDistill3D_B_t2 to saved/OpenDistill3D_B_t3..."
  cp -r saved/OpenDistill3D_B_t2 saved/OpenDistill3D_B_t3
else
  echo "saved/OpenDistill3D_B_t3 Already exists"
fi

python main_instance_segmentation.py \
general.OW_task="task3" \
general.split="B" \
general.experiment_name="OpenDistill3D_B_t3" \
general.project_name="OpenDitliss3D_train" \
general.use_conf_th=True \
general.margin=1.0 \
general.max_lr=0.0001 \
scheduler=onecyclelr_1stage \
general.use_mask_extractor=True \
general.use_scores_AL=True \
general.use_weight_func=True \
general.mask_conf_th=0.5 \
general.max_epochs=401 \
general.ucr_th=0.5


