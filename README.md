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
  6. [Erratum](#erratum)

### Installation

This code was tested on Python 2.7, Pytorch 0.4.0, CUDA 8.0
Install the dependencies with:
```Shell
pip install -r requirements.txt
```


### Data

We release data and pre-trained models for HICO-DET. To set-up the directories, please follow these steps:

1. **Download the pre-computed data** 
```Shell
wget https://www.rocq.inria.fr/cluster-willow/jpeyre/analogy/data.tar.gz
tar zxvf data.tar.gz
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
tar zxvf runs.tar.gz
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


### Erratum

Please note that the numerical results in the paper were obtained using a slightly different version for analogy transformation <img src="https://latex.codecogs.com/svg.latex?\Gamma" title="\Gamma" /> than what is described in Eq.(6) of the paper. This variant computes analogy transformation as:

<img src="https://latex.codecogs.com/svg.latex?\bm{w}^{vp}_{t'}&space;=&space;\bm{w}^{vp}_{t}&space;&plus;&space;\Gamma&space;\begin{bmatrix}&space;\bm{w}^{s}_{s'}&space;-&space;\bm{w}^{vp}_{s}&space;\\&space;\bm{w}^{p}_{p'}&space;-&space;\bm{w}^{vp}_{p}&space;\\&space;\bm{w}^{o}_{o'}&space;-&space;\bm{w}^{vp}_{o}&space;\end{bmatrix}" title="\bm{w}^{vp}_{t'} = \bm{w}^{vp}_{t} + \Gamma \begin{bmatrix} \bm{w}^{s}_{s'} - \bm{w}^{vp}_{s} \\ \bm{w}^{p}_{p'} - \bm{w}^{vp}_{p} \\ \bm{w}^{o}_{o'} - \bm{w}^{vp}_{o}, \end{bmatrix}" />

where <img src="https://latex.codecogs.com/svg.latex?\bm{w}^{s}_{s'},&space;\bm{w}^{p}_{p'},&space;\bm{w}^{o}_{o'}" title="\bm{w}^{s}_{s'}, \bm{w}^{p}_{p'}, \bm{w}^{o}_{o'}" /> are the embeddings of target subject, predicate and object in unigram spaces, and <img src="https://latex.codecogs.com/svg.latex?\bm{w}^{vp}_{s},&space;\bm{w}^{vp}_{p},&space;\bm{w}^{vp}_{o}" title="\bm{w}^{vp}_{s}, \bm{w}^{vp}_{p}, \bm{w}^{vp}_{o}" /> are the embeddings of source subject, predicate and object in visual phrase space. 

You can choose between the 2 versions through the option --analogy_type. The default option described above is called 'hybrid'. 
To run the variant described in the paper, please activate the option --analogy_type='vp' in the config file such as in './configs/hico_trainvalzeroshot_analogy_vp.yaml'.  

The variant 'vp' results in ~1% performance drop compared to the results in the paper (Table 2. s+o+vp+transfer (deep): 28.6 -> 27.5). The corresponding model is released in runs/ directory. We are still investigating why the 'hybrid' version performs better than the 'vp' one.  


We would like to thank Kenneth Wong from Institute of Computing Technology, Chinese Academy of Sciences, for his careful code review and pointing out this inconsistency. 

We apologize for this inconvenience. Also, please do not hesitate to contact the first author for further clarifications. 

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
