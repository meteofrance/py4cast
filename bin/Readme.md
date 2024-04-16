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
    - --dataset_conf  : The configuration file for the dataset 
    - --model_conf : Configuration file for the model  


In dev mode : 

    - --limit_train_batches 10 : Number of batch to use (per_gpu) 
    - --no_log : If activated, no log are kept in tensorboard. Models are not saved. 


### Examples

```sh
    runai gpu_play 1
    runai exec_gpu python bin/train.py --model hilam --dataset smeagol
```

```sh
    runai gpu_play 1
    runai exec_gpu python bin/train.py --model hilam --dataset titan
```

## 3. Playing with 'num' 'steps' options 
Currently they are many 'num' 'steps options tunable from CLI. 
 - --num_pred_steps_train : The number of prediction step for which we have some data. Used during training. 
 - --num_pred_steps_val_test : The number of prediction step for which we have some data. Used during validation and test step (from train.py). 
- --num_input_steps : Number of previous timesteps fed as inputs to the neural network
 - --num_inter_steps : Number of intermediary time steps (for which we do not have or do not use the y_true training data).
 We run num_inter_steps between each pred_step.

 This different options interfer with other options of the dataset. 

 The step_duration of the model is defined by : 

     step_duration of the dataset / num_inter_steps  

 
### Example 
 Let's take an example. 
 
 Aim : Get a model with 15 min timestep.  
 Data :  A dataset which can be used with different timestep (e.g 15 min, 1h, 3h). 

#### First step of parameters :  
 A possible set of parameter : 
    - num_input_steps : 1 
    - num_inter_steps : 1 
    - num_pred_steps_train : 1 
    - dataset : 15 min timestep.

The input will be H-15. The model will be run one time to get the prediction at time H. 

#### Second set of paramters : 
  A possible set of parameter : 
    - num_input_steps : 1 
    - num_inter_steps : 4 
    - num_pred_steps_train : 1 
    - dataset : 1h timestep.

The input will be H -1. The model will be run 4 times to get the prediction at time H. 

#### Third set of parameters : 
    - num_input_steps : 2 
    - num_inter_steps : 4 
    - num_pred_steps_train : 1 
    - dataset : 1h timestep.

The input will be H-2 and H-1. The model will be run 4 times. 
**The behavour is not the one expected. One should not used input_steps > 1 with inter_steps ? Do we reject by defaut such configuration ?**

#### Another set of parameters : 
   - num_input_steps : 1
   - num_inter_steps : 4 
   - num_pred_steps_train : 3 
   - dataset : 1h timestep 

Input will be H-1, the model will be run 4 times to obtain H, then 4 times to obtain H+1 and another 4 times to obtain H+2. 
As a consequence, there is a better computation cost with respect to dataloading. 
Is it detrimental for the minimisation ? 

#### Other information 
Note that in all those example, the metrics will be difficult to compared. 
Indeed, the dataset step_duration is not constant, which means that the cost will not be taken at the same time. 
Note that, even if we have a 15 min timestep dataset, it is not possible to have a 1h timestep for training and 15 min timestep for validation. 
However, this could be done in inference. 

Training parameters
```
    - num_input_steps : 1 
    - num_inter_steps : 4 
    - num_pred_steps_train : 1 
    - dataset : 1h timestep.
```
Inference parameters 
```
    - num_input_steps : 1 
    - num_inter_steps : 1 
    - num_pred_steps_train : 4 
    - dataset : 15 min timestep.
```

## 4. Some information on training speed. 

A few test had been conducted in order to see if there is regression in training speed due to an increase complexity. 
Here are the command launch. 
Note that the seed is fixed by default so exactly the same random number had been used.


```sh 
runai gpu_play 4
runai exec_gpu python bin/train.py --model hilam --dataset smeagol --no_log --standardize --gpu 4 --limit_train_batches 200  --batch_size 1 --step 1
runai exec_gpu python bin/train.py --model hilam --dataset smeagol --no_log --standardize --gpu 4 --limit_train_batches 200  --batch_size 1 --step 3
runai exec_gpu python bin/train.py --model hilam --dataset smeagol --no_log --standardize --gpu 1 --limit_train_batches 200  --batch_size 1 --step 1
runai exec_gpu python bin/train.py --model hilam --dataset smeagol --no_log --standardize --gpu 1 --limit_train_batches 200  --batch_size 1 --step 3
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

