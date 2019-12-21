# Detecting Unseen Visual Relations Using Analogies

Created by Julia Peyre at INRIA, Paris.

### Introduction

This is the code for the paper :

Julia Peyre, Ivan Laptev, Cordelia Schmid, Josef Sivic, Detecting Unseen Visual Relations Using Analogies, ICCV19.

The webpage for this project is available [here](http://www.di.ens.fr/willow/research/analogy/), with a link to the [paper](http://www.di.ens.fr/willow/research/analogy/paper.pdf). 

This code is available for research purpose (MIT License).  


### Contents

  1. [Installation](#installation)
  2. [Data](#data)
  3. [Train](#train)
  4. [Test](#test)
  5. [Evaluation](#evaluation)

### Installation

This code was tested on Python 2.7, Pytorch 0.4.0, CUDA 8.0


Our code requires to install the following dependencies:<br /> <br />
tensorboard\_logger <br />
opencv 2.4 <br />
pyyaml <br />
cython <br />
matplotlib <br />
scikit-image <br />
torchvision <br />

### Data

We release data and pre-trained models for HICO-DET. To set-up the directories, please follow these steps:

1. **Download the pre-computed data** 
```Shell
wget https://www.rocq.inria.fr/cluster-willow/jpeyre/analogy/data.tar.gz
unzip data.zip
```
This should be unzip into ./data folder <br />
This contains the object detections, visual features as well as database objects to run our code on HICO-DET. 

2. **Download HICO images**  
Load [HICO images](http://www-personal.umich.edu/~ywchao/hico/) and place them into directory images in ./data/hico/images :


3. **Link to COCO API** <br />
Download [COCO API](https://github.com/cocodataset/cocoapi) into new directory ./data/coco and run make 

4. **Download pre-computed models and detections**
```Shell
wget https://www.rocq.inria.fr/cluster-willow/jpeyre/analogy/runs.tar.gz
unzip runs.zip
```
This should be unzip into ./runs folder

### Train

You can re-train our model by running:

```Shell
python train.py --config_path $CONFIG_PATH
```

We provide config files in ./configs directory. <br />
Feel free to edit the config options to train variants of our model.  


### Test

You can extract the detections by running:

```Shell
python eval_hico.py --config_path $CONFIG_PATH
```

To extract the detections using our analogy model, you can run:

```Shell
python eval_hico_analogy.py --config_path $CONFIG_PATH
```


### Evaluation

We use the [official evaluation code](https://github.com/ywchao/ho-rcnn) to evaluate performance on HICO-DET 


### Cite

If you find this code useful in your research, please, consider citing our paper:

> @InProceedings{Peyre19,
>   author      = "Peyre, Julia and Laptev, Ivan and Schmid, Cordelia and Sivic, Josef",
>   title       = "Detecting Unseen Visual Relations Using Analogies",
>   booktitle   = "ICCV",
>   year        = "2019"
>}

### Questions
Any question please contact the first author julia.peyre@inria.fr
