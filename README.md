## Structure Inpainting: Image Inpainting using Structure Prediction

### Introduction:
We propose a two-stage image inpainting network which splits the task into two parts: structure reconstruction and texture generation. In the ﬁrst stage, edge-preserved smooth images are employed to train a structure reconstructor which completes the missing structures of the inputs. In the second stage, based on the reconstructed structures, a texture generator using perceptual loss is designed to yield image details.
<p align='center'>  
  <img src='https://drive.google.com/uc?export=download&id=1m11zU5-srJFuEyyTMIpE7lQctGBAl6a5' width=870/>
</p>

## Prerequisites
- Python 3
- PyTorch 1.0
- NVIDIA GPU + CUDA cuDNN

## Installation
- Clone this repo:
```bash
git clone https://github.com/kiet13/structure-inpainting.git
cd structure-inpainting
```
- Install PyTorch and dependencies from http://pytorch.org
- Install python requirements:
```bash
pip install -r requirements.txt
```

## Datasets
### 1) Images
We use [Places2](http://places2.csail.mit.edu) and [Paris Street-View](https://github.com/pathak22/context-encoder) datasets. To train a model on the full dataset, download datasets from official websites. 

After downloading, run [`scripts/flist.py`](scripts/flist.py) to generate train, test and validation set file lists. For example, to generate the training set file list on Places2 dataset run:
```bash
mkdir datasets
python ./scripts/flist.py --path path_to_places2_train_set --output ./datasets/places_train.flist
```

### 2) Irregular Masks
Our model is trained on the irregular mask dataset provided by [Liu et al.](https://arxiv.org/abs/1804.07723). You can download publically available Irregular Mask Dataset from [their website](http://masc.cs.gmu.edu/wiki/partialconv).

## Getting Started
Download the pre-trained models using the following links and copy them under `./checkpoints` directory. Those are trained on G'1 (StructureFlow) generator in first stage.

[Places2](https://drive.google.com/drive/folders/1ZEo5jB3AVbYURBDXNJblGJYZYlZf4XGM?usp=sharing) | [Paris-StreetView](https://drive.google.com/drive/folders/10w9_RsqDHyEM-f4RyBt1pMiDDDZWaVOl?usp=sharing)


### 1) Training
To train the model, create a `config.yaml` file similar to the [example config file](https://github.com/knazeri/edge-connect/blob/master/config.yml.example) and copy it under your checkpoints directory.

Structure Inpainting is trained in three stages: 1) training the structure model, 2) training the inpaint model and 3) training the joint model. To train the model:
```bash
python train.py --model [stage] --checkpoints [path to checkpoints]
```

For example to train the structure model on Places2 dataset under `./checkpoints/places2` directory:
```bash
python train.py --model 1 --checkpoints ./checkpoints/places2
```

### 2) Testing
To test the model, create a `config.yaml` file similar to the [example config file](config.yml.example) and copy it under your checkpoints directory.

You can test the model on all three stages: 1) structure model, 2) inpaint model and 3) joint model. In each case, you need to provide an input image (image with a mask) and a grayscale mask file. Please make sure that the mask file covers the entire mask region in the input image. To test the model:
```bash
python test.py \
  --model [stage] \
  --checkpoints [path to checkpoints] \
  --input [path to input directory or file] \
  --struct [path to structure directory or file] \
  --mask [path to masks directory or mask file] \
  --output [path to the output directory]
```

We provide some test examples under `./examples` directory. Please download the [pre-trained models](#getting-started) and run:
```bash
python test.py \
  --checkpoints ./checkpoints/places2 
  --input ./examples/places2/images
  --struct ./examples/places2/structs
  --mask ./examples/places2/masks
  --output ./checkpoints/results
```
This script will inpaint all images in `./examples/places2/images` using their corresponding masks in `./examples/places2/masks` directory and structure in `./examples/places2/structs' saves the results in `./checkpoints/results` directory.


### Structure Generation
We do not apply any image smoothing technique in default. If you want to train the model with certain smoothing technique, you need to generate structure maps for entire training/test sets as a pre-processing and their corresponding file lists using [`scripts/flist.py`](scripts/flist.py). Please make sure the file names and directory structure match your training/test sets. In this project, we use [`L0`](http://www.cse.cuhk.edu.hk/~leojia/projects/L0smoothing/) and [`SGF`](https://github.com/feihuzhang/SGF) as a smoothing technique for structure generation model.

#### Generate smoothing image:
```bash
chmod 755 ./scripts/gen_structure.sh
./scripts/gen_structure.sh input_folder output_folder smoothing (L0/SGF) 
```

### Model Configuration

The model configuration is stored in a [`config.yaml`](config.yml.example) file under your checkpoints directory. The following tables provide the documentation for all the options available in the configuration file:

#### General Model Configurations

Option          | Description
----------------| -----------
MODE            | 1: train, 2: test, 3: eval
MODEL           | 1: structure model, 2: inpaint model, 3: joint model
APOS            | 0: G1 (EdgeConnect), 1: G'1 (StructureFlow)
MASK            | 1: random block, 2: half, 3: external, 4: external + random block, 5: external + random block + half
SEED            | random number generator seed
GPU             | list of gpu ids, comma separated list e.g. [0,1]
DEBUG           | 0: no debug, 1: debugging mode
VERBOSE         | 0: no verbose, 1: output detailed statistics in the output console

#### Loading Train, Test and Validation Sets Configurations

Option          | Description
----------------| -----------
TRAIN_FLIST     | text file containing training set files list
VAL_FLIST       | text file containing validation set files list
TEST_FLIST      | text file containing test set files list
TRAIN_STRUCT_FLIST| text file containing training set external structure files list
VAL_STRUCT_FLIST  | text file containing validation set external structure files list
TEST_STRUCT_FLIST | text file containing test set external structure files list
TRAIN_MASK_FLIST| text file containing training set masks files list (only with MASK=3, 4, 5)
VAL_MASK_FLIST  | text file containing validation set masks files list (only with MASK=3, 4, 5)
TEST_MASK_FLIST | text file containing test set masks files list (only with MASK=3, 4, 5)

#### Training Mode Configurations

Option                 |Default| Description
-----------------------|-------|------------
LR                     | 0.0001| learning rate
D2G_LR                 | 0.1   | discriminator/generator learning rate ratio
BETA1                  | 0.5   | adam optimizer beta1
BETA2                  | 0.9   | adam optimizer beta2
BATCH_SIZE             | 8     | input batch size 
INPUT_SIZE             | 256   | input image size for training. (0 for original size)
MAX_ITERS              | 2e6   | maximum number of iterations to train the model
INPAINT_L1             | 1     | l1 loss weight for G2
STRUCTURE_L1           | 4    | l1 loss weight for G1
STYLE_LOSS_WEIGHT      | 250     | style loss weight
CONTENT_LOSS_WEIGHT    | 0.1     | perceptual loss weight
INPAINT_ADV_LOSS_WEIGHT| 0.1  | adversarial loss weight
GAN_LOSS               | nsgan | **nsgan**: non-saturating gan, **lsgan**: least squares GAN, **hinge**: hinge loss GAN
GAN_POOL_SIZE          | 0     | fake images pool size
SAVE_INTERVAL          | 1000  | how many iterations to wait before saving model (0: never)
EVAL_INTERVAL          | 0     | how many iterations to wait before evaluating the model (0: never)
LOG_INTERVAL           | 10    | how many iterations to wait before logging training loss (0: never)
SAMPLE_INTERVAL        | 1000  | how many iterations to wait before saving sample (0: never)
SAMPLE_SIZE            | 10    | number of images to sample on each samling interval

### Acknowledgements

We built our code based on [Edge-Connect](https://github.com/knazeri/edge-connect). Please consider to cite their papers. 
