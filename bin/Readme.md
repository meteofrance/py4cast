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


## 3. Some information on training speed. 

A few test had been conducted in order to see if there is regression in training speed due to an increase complexity. 
Here are the command launch. 
Note that the seed is fixed by default so exactly the same random number had been used.


```sh 
runai gpu_play 4
runai exec_gpu python bin/train.py --model graph --dataset smeagol --no_log --standardize --gpu 4 --limit_train_batches 200  --batch_size 1 --step 1
runai exec_gpu python bin/train.py --model graph --dataset smeagol --no_log --standardize --gpu 4 --limit_train_batches 200  --batch_size 1 --step 3
runai exec_gpu python bin/train.py --model graph --dataset smeagol --no_log --standardize --gpu 1 --limit_train_batches 200  --batch_size 1 --step 1
runai exec_gpu python bin/train.py --model graph --dataset smeagol --no_log --standardize --gpu 1 --limit_train_batches 200  --batch_size 1 --step 3
```

NB : The it per second is increasing batch after batch. There seem to be an initial cost which vanish. 


** Other important factors  which may impact a lot : **
  - num_workers : 10
  - prefetch : 2
  - Grid: 500 x 500


|  | 1 Step | 3 Steps |
|--|--|--|
|1 GPU | 1.53 it/s -> 2:10 | 0.59 it/s -> 5:36 |
|4 GPU | 0.78 it/s -> 1:04 | 0.44 it/s -> 1:54 |


Test conducted on MR !14 

|  | 1 Step | 3 Steps |
|--|--|--|
|1 GPU |                   | 0.59 it/s -> 5:39|
|4 GPU | 0.80 it/s -> 1:02 | 0.43 it/s -> 1:55| 


Test conducted on MR !26 

- Grid 512 x 512
- prefetch : None

The way to select the number of batch has changed. Previously we globally limit the number of batch. Now it's a limit per gpu. 

|  | 1 Step | 3 Steps |
|--|--|--|
|1 GPU | 1.51 it/s -> 2:12 | 0.58 it/s -> 5:42  |
|4 GPU | 0.88 it/s -> 3:48 |  | 

## 4. Other informations
### 4.1 Memory Leak
The script `bin/minimal_leak.py` illustrate some memory leaks using pytorch/xarray on a docker container. 
This investigation had been carried out after a OOM error due to smeagol dataLoader. 

Figure ![Alt text](../doc/figs/memory_leak.png) shows the memory consumption on several cases. 
One can see that when doing the normalisation in pytorch, opening a netcdf file using xarray leads to a big leak. 
This is no longer the case when we do not open a file (but in real world, no data will come in) and the leak is limited if normalisation 
is done in numpy. 


**Results are not reproducible** : The behaviour (e.g for with open_xarray()) does not provides the same curve each time it is launched (even if we use the same node).

