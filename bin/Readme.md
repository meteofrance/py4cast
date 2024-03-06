# How to launch a training ?

## 1. Prepare the dataset and the model chosed

To be explained.

```bash
runai exec_gpu python bin/prepare.py titan all
```

```bash
runai exec_gpu python bin/prepare.py nlam --dataset titan
```

## 2. Launch the training (via training.py)

Main options are :

    - --model  ["hi_lam","graph_lam"] : The model choosed
    - --dataset ["titan","smeagol"] : The dataset choosed
    - --data_conf  : The configuration file for the dataset (used only for smeagol right now).
    - --steps : Number of autoregressive steps 
    - --standardize : Do we want to standardize our inputs ? 

In dev mode : 

    - --subset 10 : Number of batch to use 
    - --no_log : If activated, no log are kept in tensorboard. Models are not saved. 


### Examples

```sh
    runai gpu_play_mono
    runai exec_gpu python bin/train.py --model hi_lam --dataset smeagol
```

```sh
    runai gpu_play_mono
    runai exec_gpu python bin/train.py --model hi_lam --dataset titan
```