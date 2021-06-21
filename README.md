## Structure Inpainting: Image Inpainting using Structure Prediction

### Introduction:
We develop a new approach for image inpainting that does a better job of reproducing filled regions exhibiting fine details inspired by our understanding of how artists work: *lines first, color next*. We propose a two-stage adversarial model EdgeConnect that comprises of an edge generator followed by an image completion network. The edge generator hallucinates edges of the missing region (both regular and irregular) of the image, and the image completion network fills in the missing regions using hallucinated edges as a priori. Detailed description of the system can be found in our [paper](https://arxiv.org/abs/1901.00212).
<p align='center'>  
  <img src='https://user-images.githubusercontent.com/1743048/50673917-aac15080-0faf-11e9-9100-ef10864087c8.png' width='870'/>
</p>
(a) Input images with missing regions. The missing regions are depicted in white. (b) Computed edge masks. Edges drawn in black are computed (for the available regions) using Canny edge detector; whereas edges shown in blue are hallucinated by the edge generator network. (c) Image inpainting results of the proposed approach.

## Prerequisites
- Python 3
- PyTorch 1.0
- NVIDIA GPU + CUDA cuDNN

## Installation
- Clone this repo:
```bash
git clone https://github.com/knazeri/edge-connect.git
cd edge-connect
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

Alternatively, you can download [Quick Draw Irregular Mask Dataset](https://github.com/karfly/qd-imd) by Karim Iskakov which is combination of 50 million strokes drawn by human hand.

Please use [`scripts/flist.py`](scripts/flist.py) to generate train, test and validation set masks file lists as explained above.

## Getting Started
Download the pre-trained models using the following links and copy them under `./checkpoints` directory.

[Places2](https://drive.google.com/drive/folders/1KyXz4W4SAvfsGh3NJ7XgdOv5t46o-8aa) | [Paris-StreetView](https://drive.google.com/drive/folders/1cGwDaZqDcqYU7kDuEbMXa9TP3uDJRBR1)


### 1) Training
To train the model, create a `config.yaml` file similar to the [example config file](https://github.com/knazeri/edge-connect/blob/master/config.yml.example) and copy it under your checkpoints directory. Read the [configuration](#model-configuration) guide for more information on model configuration.

Structure Inpainting is trained in three stages: 1) training the structure model, 2) training the inpaint model and 3) training the joint model. To train the model:
```bash
python train.py --model [stage] --checkpoints [path to checkpoints]
```

For example to train the edge model on Places2 dataset under `./checkpoints/places2` directory:
```bash
python train.py --model 1 --checkpoints ./checkpoints/places2
```

Convergence of the model differs from dataset to dataset. For example Places2 dataset converges in one of two epochs, while smaller datasets like CelebA require almost 40 epochs to converge. You can set the number of training iterations by changing `MAX_ITERS` value in the configuration file.

### 2) Testing
To test the model, create a `config.yaml` file similar to the [example config file](config.yml.example) and copy it under your checkpoints directory. Read the [configuration](#model-configuration) guide for more information on model configuration.

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


### Alternative Structure Generation
We do not apply any image smoothing technique in default. If you want to train the model with certain smoothing technique, you need to generate structure maps for entire training/test sets as a pre-processing and their corresponding file lists using [`scripts/flist.py`](scripts/flist.py). Please make sure the file names and directory structure match your training/test sets. In this project, we use L_0 and SGF as a smoothing technique for G1.

### Model Configuration

The model configuration is stored in a [`config.yaml`](config.yml.example) file under your checkpoints directory. The following tables provide the documentation for all the options available in the configuration file:

#### General Model Configurations

Option          | Description
----------------| -----------
MODE            | 1: train, 2: test, 3: eval
MODEL           | 1: edge model, 2: inpaint model, 3: edge-inpaint model, 4: joint model
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
TRAIN_STRUCT_FLIST| text file containing training set external edges files list
VAL_STRUCT_FLIST  | text file containing validation set external edges files list
TEST_STRUCT_FLIST | text file containing test set external edges files list
TRAIN_MASK_FLIST| text file containing training set masks files list (only with MASK=3, 4, 5)
VAL_MASK_FLIST  | text file containing validation set masks files list (only with MASK=3, 4, 5)
TEST_MASK_FLIST | text file containing test set masks files list (only with MASK=3, 4, 5)

#### Training Mode Configurations

Option                 |Default| Description
-----------------------|-------|------------
LR                     | 0.0001| learning rate
D2G_LR                 | 0.1   | discriminator/generator learning rate ratio
BETA1                  | 0.0   | adam optimizer beta1
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
