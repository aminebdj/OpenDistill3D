### Download pre-trained models
We provide in <a href="https://mbzuaiac-my.sharepoint.com/:u:/g/personal/mohamed_boudjoghra_mbzuai_ac_ae/EfQ13YdGk_tIhT6dfxTNiPEBu6YyfdahULbORc8K3643tA?e=ByNBQ4">this link</a> the closed-setting pre trained Mask3D model to initialize the teacher models for the three splits.

Download the `checkpoints.zip` and place them inside `./mask_extractor`. The code structure should be as follows

<b>NB:</b> models in `checkpoints.zip` follow Mask3D architecture and are trained in a closed setting manner on the classes known in Task1.


```
├── benchmark
├── conf                                 <- hydra configuration files
├── datasets
│   ├── preprocessing                    <- folder with preprocessing scripts
│   ├── semseg.py                        <- indoor dataset
│   └── utils.py
├── docs
├── mask_extractor                       <- for the teacher model
│   ├── checkpoints                      <- has the checkpoints to initilize the teacher model in Task 1
│   ├── configs                          <- folder with config of the teacher model
│   ├── AutoLabeler.py                   <- teacher model
│   ├── __init__.py
│   └── mask3d.py.py                     <- teacher model architecture
├── models                               <- OpenDistlill3D model, with Mask3D modules
├── data
│   ├── processed                        <- folder for preprocessed datasets
│   └── raw                              <- folder for raw datasets
├── scripts                              <- train scripts
├── third_party                          <- third party, for Minkowski engine
├── trainer
│   ├── __init__.py
│   └── trainer.py                       <- train loop
├── utils
├── Visualization
├── README.md
├──saved                                 <- folder that stores models and logs
├──README.md                             <- readme file
├──evaluate.sh                           <- evaluation script
└──main_instance_segmentation.py         <- the main file
```


### Conda environment setup
The main dependencies of the project are the following:
```yaml
python: 3.10.6
cuda: 11.6
```
You can set up a conda environment as follows
```
conda create --name=opendistill3d python=3.10.6
conda activate opendistill3d

conda update -n base -c defaults conda
conda install openblas-devel -c anaconda

pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.12.1+cu116.html

pip install ninja==1.10.2.3
pip install pytorch-lightning==1.7.2
pip install fire imageio tqdm wandb python-dotenv pyviz3d scipy plyfile scikit-learn trimesh loguru albumentations

pip install "cython<3.0.0" wheel && pip install pyyaml==5.4.1 --no-build-isolation
pip install volumentations==0.1.8

pip install antlr4-python3-runtime==4.8
pip install black==21.4b2
pip install omegaconf==2.0.6 hydra-core==1.0.5 --no-deps
pip install 'git+https://github.com/facebookresearch/detectron2.git@710e7795d0eeadf9def0e7ef957eea13532e34cf' --no-deps

cd third_party/pointnet2 && python setup.py install

pip install fvcore
pip install reliability
pip install shortuuid
pip install pycocotools==2.0.7
pip install seaborn 
pip install cloudpickle==2.1.0

git clone https://github.com/NVIDIA/MinkowskiEngine.git
cd MinkowskiEngine
python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas

```

### Data preprocessing
After installing the dependencies, we preprocess the datasets.

#### ScanNet200
First, we apply Felzenswalb and Huttenlocher's Graph Based Image Segmentation algorithm to the test scenes using the default parameters.
Please refer to the [original repository](https://github.com/ScanNet/ScanNet/tree/master/Segmentator) for details.
Put the resulting segmentations in `./data/raw/scannet_test_segments`.
```
python datasets/preprocessing/scannet_preprocessing.py preprocess \
--data_dir="PATH_TO_RAW_SCANNET_DATASET" \
--save_dir="../../data/processed/scannet200" \
--git_repo="PATH_TO_SCANNET_GIT_REPO" \
--scannet200=true
```

