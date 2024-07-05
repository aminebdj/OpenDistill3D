import os
import yaml
import numpy as np
import open3d as o3d
import random
from datasets.scannet200.owis_splits import KNOWN_CLASSES_LABELS,UNKNOWN_CLASSES_LABELS
from datasets.scannet200.scannet200_constants import CLASS_LABELS_200,VALID_CLASS_IDS_200
from copy import copy
# OUTPUT_2 = "output-scans/predictions-ours-2classes/"
label2id = {l:id for id,l in zip(VALID_CLASS_IDS_200, CLASS_LABELS_200)}
def generate_color_palette(x):
    palette = []

    while len(palette) < x:
        r, g, b = random.sample(range(256), 3)
        color = (r/255.0, g/255.0, b/255.0)
        if color[2] > 0:
            continue
        palette.append(color)

    return palette

def read_ply_file(file_path):
    # Read the ply file
    point_cloud = o3d.io.read_triangle_mesh(file_path)
    return point_cloud

model_name = "experiment_name"

label_database = "data/processed/scannet200/label_database.yaml"

with open(label_database) as f:
    x = yaml.safe_load(f) 


SCENE  = "scene0000_00"
split = ""
task = ""
# read label database
cls_ids = x.keys()
# from dict keys to list
cls_ids = [int(i) for i in cls_ids]
OUTPUT_gt = f"./{SCENE}_gt.ply"
OUTPUT_in = f"./{SCENE}_in.ply"
pc = []
pco = []
# if SCENE in good_scenes:
PATH_PLY = f"data/raw/scannet_test_segments/scans/{SCENE}/{SCENE}_vh_clean_2.labels.ply"
PATH_PLY_1 = f"data/raw/scannet_test_segments/scans/{SCENE}/{SCENE}_vh_clean_2.ply"

# read label database
try:
    label = f"data/processed/scannet200/instance_gt/validation/{SCENE}.txt"
    # read label file
    with open(label) as f:
        lines = f.readlines()
    unique_ids = np.unique(np.asarray(lines))
    unique_ids = [int(i[:-1]) for i in unique_ids]
    classes = [i//1000 for i in unique_ids if i !=0]
    # if task != 'task3':
    if UNKNOWN_CLASSES_LABELS[split][task] != None:
        UNKNOWNS =  [label2id[i] for i in UNKNOWN_CLASSES_LABELS[split][task]]
    else:
        UNKNOWNS = []
        
    instances = []

    

    pc = read_ply_file(PATH_PLY)
    pco = read_ply_file(PATH_PLY_1)
    points = np.asarray(pc.vertices)
    indices = []
    for instance in instances:
        path = instance.split(" ")[0].strip()
        cls = instance.split(" ")[1].strip()

       
            
        pc = read_ply_file(PATH_PLY)
        points = np.asarray(pc.vertices)
        print(cls)
        continue
        for i in range(points.shape[0]):
        # change color of point based on its instance id
            
            if(int(lines[i][:-1])//1000 == 0 or int(lines[i][:-1])//1000 == 3 or int(lines[i][:-1])//1000 == 1): # make gray if 0 (nothing) or 3 (floor) or 1 (wall)
                pc.vertex_colors[i] = [0.5, 0.5, 0.5]
            elif(int(lines[i][:-1])//1000 in UNKNOWNS):
                pc.vertex_colors[i] = [31/255,119/255,180/255]#[23/255,124/255,172/255]
            else:
                pc.vertex_colors[i] = [42/255,157/255,37/255] # green
        o3d.io.write_triangle_mesh(os.path.join(OUTPUT_gt, SCENE + "-gt.ply"), pc)
        o3d.io.write_triangle_mesh(os.path.join(OUTPUT_in, SCENE + "-input.ply"), pco)
except:
    pass