# How to launch a training ?  
## 1. Prepare the dataset and the model chosed 
To be explained. 

## 2. Launch the training (via training_vincent.py)
Main options are : 
    - --model ["hi_lam","graph_lam"] : The model choosed 
    - --dataset ["titan","smeagol"] : The dataset choosed 
    - --data_conf  : The configuration file for the dataset (used only for smeagol right now).

### Examples  
    ```sh
    runai gpu_play_mono 
    runai exec_gpu python bin/train_vincent.py --model hi_lam --dataset smeagol  
    ```
    ```sh
    runai gpu_play_mono 
    runai exec_gpu python bin/train_vincent.py --model hi_lam --dataset titan  
    ```