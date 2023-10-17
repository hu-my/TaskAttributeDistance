<<<<<<< HEAD
# TaskAttributeDistance
=======
# Understanding Few-Shot Learning: Measuring Task Relatedness and Adaptation Difficulty via Attributes 
This repository is the official implementation of "Understanding Few-Shot Learning: Measuring Task Relatedness and Adaptation Difficulty via Attributes".

## Dependenices

The code is built with following libraries:
- python 3.7
- PyTorch 1.7.1
- cv2
- matplotlib
- sklearn
- tensorboard
- h5py
- tqdm

#### Installation
```setup
conda create -n TAD python=3.7
source activate TAD
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
pip install -r requirements.txt
```

#### Dataset prepare
Please download the CUB and SUN datasets, then put them under the path of `filelists/<dataset name>/`.

Here we provide a [link](https://drive.google.com/file/d/1Je-BZaCVe9fSoUUpkBhBlm8thxalRxkI/view?usp=sharing) of CUB dataset and related files.

>📋  Describe how to set up the environment, e.g. pip/conda/docker commands, download datasets, etc...

## Training

To train the FSL models (such as ProtoNet) on CUB dataset, run this command:

```train
bash scripts/train/cub_protonet.sh
```

## Evaluation

To evaluate models on CUB, run:

```eval
bash scripts/test/cub_protonet.sh
```

## Plot task distance and accuracy

To estimate the average TAD between each novel task and training tasks, then plot a figure of average TAD and accuracy, run:

```eval
bash scripts/test/plot_distance_acc.sh
```

## Pre-trained Models

Here we provide some pretrained models for fast strat.

- [ProtoNet (Conv4NP)](https://drive.google.com/file/d/1AxXRP0QSmH0C5Y3i8GXEHThg6otK8leH/view?usp=sharing) trained on CUB in the 5-way 1-shot setting 

Download the pretrained model at file path `checkpoints/CUB/Conv4NP_protonet_0_aug_5way_1shot/`, and then run the command in `Plot task distance and accuracy` part.

Our codebase is developed based on the [baseline++](https://github.com/wyharveychen/CloserLookFewShot) from the paper [A Closer Look at Few-shot Classification](https://arxiv.org/abs/1904.04232) and [COMET](https://github.com/snap-stanford/comet) from the paper [Concept Learners for Few-Shot Learning](https://arxiv.org/pdf/2007.07375.pdf).
>>>>>>> 4aa4d55... initial commit
