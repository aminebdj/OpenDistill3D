import logging
import os
import hydra 
from dotenv import load_dotenv
from trainer.trainer import InstanceSegmentation, RegularCheckpointing
from omegaconf import DictConfig, OmegaConf
from utils.utils import (
    flatten_dict, 
    load_checkpoint_with_missing_or_exsessive_keys,
    load_backbone_checkpoint_with_missing_or_exsessive_keys
)
from pytorch_lightning import Trainer, seed_everything
from omegaconf import OmegaConf

import torch

from datasets.scannet200.scannet200_constants import CLASS_LABELS_200

CLASS_LABELS_200 = list(CLASS_LABELS_200)
CLASS_LABELS_200.remove('floor')
CLASS_LABELS_200.remove('wall')
MAP_STRING_TO_ID = {CLASS_LABELS_200[i] : i for i in range(len(CLASS_LABELS_200))}
MAP_STRING_TO_ID['background'] = 253

def get_parameters(cfg: DictConfig):
    
    logger = logging.getLogger(__name__)
    load_dotenv(".env")

    # parsing input parameters
    seed_everything(cfg.general.seed)

    # getting basic configuration 
    if cfg.general.get("gpus", None) is None:
        cfg.general.gpus = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    loggers = []

    if not os.path.exists(cfg.general.save_dir):
        os.makedirs(cfg.general.save_dir)
    else:
        print("EXPERIMENT ALREADY EXIST")
        cfg['trainer']['resume_from_checkpoint'] = f"{cfg.general.save_dir}/last-epoch.ckpt"

    for log in cfg.logging:
        print(log)
        
        loggers.append(hydra.utils.instantiate(log))
        loggers[-1].log_hyperparams(
            flatten_dict(OmegaConf.to_container(cfg, resolve=True))
        )
    
    model = InstanceSegmentation(cfg)
    if cfg.general.backbone_checkpoint is not None:
        cfg, model = load_backbone_checkpoint_with_missing_or_exsessive_keys(cfg, model)
    if cfg.general.checkpoint is not None:
        cfg, model = load_checkpoint_with_missing_or_exsessive_keys(cfg, model)
    
    logger.info(flatten_dict(OmegaConf.to_container(cfg, resolve=True)))
    return cfg, model, loggers


@hydra.main(config_path="conf", config_name="config_base_instance_segmentation.yaml")
def train(cfg: DictConfig):
    if cfg.general.OW_task != "task1" and cfg.general.finetune:
        cfg.general.save_dir = cfg.general.save_dir+"_finetune"
        cfg.general.logg_suffix = cfg.general.logg_suffix+"_finetune"
    os.chdir(hydra.utils.get_original_cwd())
    cfg, model, loggers = get_parameters(cfg)
    callbacks = [] 
    for cb in cfg.callbacks:
        callbacks.append(hydra.utils.instantiate(cb))

    callbacks.append(RegularCheckpointing())
    runner = Trainer(
        logger=loggers,
        gpus=cfg.general.gpus,
        callbacks=callbacks,
        weights_save_path=str(cfg.general.save_dir),
        **cfg.trainer
    )

    if not cfg.general.train_oracle and cfg.general.train_mode:
        if cfg.general.OW_task != "task1" and (not cfg.general.finetune):
            task = cfg.general.save_dir.split('/')[-1]
            prev_task = task.replace(task[-1], str(int(task[-1])-1))
            if os.path.exists(cfg.general.save_dir.replace(task, prev_task)):
                list_dir = os.listdir(cfg.general.save_dir.replace(task, prev_task))
                print(list_dir)
                for file in list_dir:
                    file_s = file.split("_")
                    if "ap" in file_s:
                        path_to_prev_task = cfg.general.save_dir.replace(task, prev_task)+"/"+file
                        break
                    else:
                        path_to_prev_task = None
                    
                if path_to_prev_task == None:
                    print(f"There is no best model in {prev_task} ❌")
                    exit()
                                          
            checkpoint = torch.load(path_to_prev_task)
            if not os.path.exists(cfg.general.save_dir+"/"+"last-epoch.ckpt"):
                model.load_state_dict(checkpoint["state_dict"])
            if (list(model.state_dict().values())[0] == list(checkpoint["state_dict"].values())[0].to(model.device)).all():
                print(f"{path_to_prev_task} was loaded successfuly ✅")
            elif (not os.path.exists(cfg.general.save_dir+"/"+"last-epoch.ckpt")) and (not ((list(model.state_dict().values())[0] == list(checkpoint["state_dict"].values())[0].to(model.device)).all())):
                print(f"Failed loadng model {path_to_prev_task} ❌")
                exit()
            else:
                print(f"Experiment for {task} already exists")
                
        elif cfg.general.OW_task != "task1" and cfg.general.finetune:
            resume_from = os.path.join(cfg.general.save_dir.replace("_finetune",""),"last.ckpt")
            checkpoint = torch.load(resume_from)
            if not os.path.exists(cfg.general.save_dir+"/"+"last.ckpt"):
                model.load_state_dict(checkpoint["state_dict"]) 
    runner.fit(model)
    
@hydra.main(config_path="conf", config_name="config_base_instance_segmentation.yaml")
def test(cfg: DictConfig):
    os.chdir(hydra.utils.get_original_cwd())
    cfg, model, loggers = get_parameters(cfg)
    runner = Trainer(
        gpus=cfg.general.gpus,
        logger=loggers,
        weights_save_path=str(cfg.general.save_dir),
        **cfg.trainer
    )
    runner.test(model)

@hydra.main(config_path="conf", config_name="config_base_instance_segmentation.yaml")
def main(cfg: DictConfig):
    print(cfg.general.save_dir)
    if cfg['general']['train_mode']:
        train(cfg)
    else:
        test(cfg)
        

if __name__ == "__main__":    main()
 