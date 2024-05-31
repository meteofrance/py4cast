# PY4CAST

This project, built using **PyTorch** and **PyTorch-lightning**, is designed to train a variety of Neural Network architectures (GNNs, CNNs, Vision Transformers, ...) on various weather forecasting datasets. This is a **Work in Progress**, intended to share ideas and design concepts with partners.

Developped at Météo-France by **DSM/AI Lab** and **CNRM/GMAP/PREV**.

Contributions are welcome.

This project is licensed under the [APACHE 2.0 license.](LICENSE-2.0.txt)

# Acknowledgements

This project started as a fork of neural-lam, a project by Joel Oskarsson, see [here](https://github.com/mllam/neural-lam). Many thanks to Joel for his work!

# Contacts

Feel free to contact us at: <lea.berthomier@meteo.fr>, <vincent.chabot@meteo.fr>, <frank.guibert@meteo.fr>

# Table of contents

0. [Overview](#overview)
1. [Features](doc/features.md)
    1. [Neural network architectures](doc/features.md#available-pytorchs-architectures)
    2. [Datasets](doc/features.md#available-datasets)
    3. [Losses](doc/features.md#available-losses)
    4. [Plots](doc/features.md#available-plots)
    5. [Training strategies](doc/features.md#available-training-strategies)
    6. [NamedTensors](doc/features.md#namedtensors)
2. [Installation](#installation)
3. [Usage](#usage)
    1. [Docker and runai (MF)](#docker-and-runai)
    2. [Conda or Micromamba](#conda-or-micromamba)
    3. [Specifying your sbatch card](#specifying-your-sbatch-card)
    4. [Dataset configuration & simple training](#dataset-configuration--simple-training)
    5. [Training options](#training-options)
    6. [Experiment tracking](#tracking-experiment)
4. [Contributing new features](#adding-features-and-contributing)
    1. [Adding a new neural network architecture](doc/add_features_contribute.md#adding-a-new-neural-network-architecture-to-the-project)
    2. [Adding a new dataset](doc/add_features_contribute.md#adding-a-new-dataset)
    3. [Adding plots](doc/add_features_contribute.md#adding-training-plots)
5. [Design choices](#design-choices)
6. [Unit tests](doc/add_features_contribute.md#unit-tests)
7. [Continuous Integration](doc/add_features_contribute.md#continuous-integration)


## Overview

* 5 neural network architectures : Half-Unet, U-Net, SegFormer, HiLam, GraphLam
* 1 dataset with samples available on Huggingface : Titan
* 2 training strategies : Scaled Auto-regressive steps, Differential Auto-regressive steps
* 4 losses: Scaled RMSE, Scaled L1, Weighted MSE, Weighted L1
* neural networks as simple torch.nn.Module
* training with pytorchlightning
* simple interfaces to easily add a new dataset, neural network, training strategy or loss
* simple command line to lauch a training
* config files to change the parameters of your dataset or neural network during training
* experiment tracking with tensorboard and plots of forecasts with matplotlib
* implementation of [NamedTensors](doc/features.md/#namedtensors) to tracks features and dimensions of tensors at each step of the training

See [here](doc/features.md) for details on the available datasets, neural networks, training strategies, losses, and explanation of our NamedTensor.

## Installation

Start by cloning the repository:
```bash
git clone https://github.com/meteofrance/py4cast.git
cd py4cast
```

### At Météo-France

Wehn working at Météo-France, you can use either runai + Docker or Conda/Micromamba to setup a working environment. On the AI Lab cluster we recommend using runai, Conda on our HPC.

See the [runai repository](https://git.meteo.fr/dsm-labia/monorepo4ai) for installation instructions.

### Install with micromamba

Please install the environment using :
```sh
micromamba create -f env.yaml
```

Depending on your machine, you may have to run a specific command to be able to use the GPUs. For instance, on a machine with `slurm`, you could run a classic srun command to get an interactive shell on gpu:

```sh
srun --job-name="MySuperJob" --cpus-per-task=15 --partition=node3 --gres=gpu:v100:1 --time=10:00:00 --ntasks-per-node=1 --pty bash
```

Activate your environnment :
```sh
micromamba activate nlam
```

Go to py4cast directory and do :
```sh
export PYTHONPATH=`pwd`
```

You can also add directly the directory in your `.bashrc`.

This is done in order to register the py4cast package in the python path.

### Install with conda

You can install a conda environment using
```sh
conda env create --file env_conda.yaml
```
Note that this environment is a bit different from the micromamba one. Both should be merged in a near future.


### Setting environment variables

In order to be able to run the code on different machines, some environment variables can be set.
You may add them in your `.bashrc` or modify them just before launching an experiment.

- `PY4CAST_ROOTDIR` : Specify the ROOT DIR for your experiment. It also modifies the CACHE_DIR.
- `PY4CAST_SMEAGOL_PATH`: Specify where the smeagol dataset is stored. Only needed if you want to use the smeagol dataset. **Should be moved in the configuration file**.
- `PY4CAST_TITAN_PATH`: Specify where the titan dataset is stored. Only needed if you want to use the titan dataset.
If you plan to use micromamba or conda you should also add `py4cast` to your **PYTHONPATH** by expanding it (Export or change your `PYTHONPATH`).

## Usage

### Docker and runai

For now this works only for internal Météo-France users.

<details>
<summary>Click to expand</summary>

`runai` commands must be issued at the root directory of the `py4cast` project:

1. Run an interactive training session

```bash
runai gpu_play 4
runai build
runai exec_gpu python bin/train.py --dataset titan --model hilam
```

2. Train using sbatch single node multi-GPUs

```bash
export RUNAI_GRES="gpu:v100:4"
runai sbatch python bin/train.py --dataset titan --model hilam
```

3. Train using sbatch multi nodes multi GPUs

Here we use 2 nodes with 4 GPUs each.

```bash
export RUNAI_SLURM_NNODES=2
export RUNAI_SLURM_NTASKS_PER_NODE=4
export RUNAI_GRES="gpu:v100:4"
runai sbatch_multi_node python bin/train.py --dataset titan --model hilam
```

For the rest of the documentation, you must preprend each python command with `runai exec_gpu`.
</details>


### Conda or Micromamba

Once your micromamba environment is setup, you can do a classical training such as
```sh
python3.10 bin/train.py  --dataset titan --model halfunet
```

For the rest of the documentation, you must replace each python command with `python3.10`.

### Specifying your sbatch card

To do so, you will need a small `sh` script.

```sh
#!/usr/bin/bash
#SBATCH --partition=ndl
#SBATCH --nodes=1 # Specify the number of GPU node you required
#SBATCH --gres=gpu:1 # Specify the number of GPU required per Node
#SBATCH --time=05:00:00 # Specify your experiment Time limit
#SBATCH --ntasks-per-node=1 # Specify the number of task per node. This should match the number of GPU Required per Node

# Note that other variable could be set (according to your machine). For example you may need to set the number of CPU or the memory used by your experiment.
# On MF hpc, this is proportional to the number of GPU required per node. This is not the case on other machine (e.g MétéoFrance AILab machine).

source ~/.bashrc  # Be sure that all your environment variables are set
conda activate py4cast # Activate your environment (installed by micromamba or conda)
cd $PY4CAST_PATH # Go to Py4CAST (you can either add an environment variable or hard code it here)
# Launch your favorite command.
srun python bin/train.py --model halfunet --dataset smeagol --dataset_conf config/smeagolsmall.json --num_pred_steps_val_test 4 --strategy scaled_ar --num_inter_steps 4 --num_input_steps 1 --batch_size 10
```

Then just launch this script using

```sh
sbatch my_tiny_script.sh
```

### Dataset configuration & simple training

As in neural-lam, before training you must first compute the mean and std of each feature.

To compute the stats of the Titan dataset:

```bash
python py4cast/datasets/titan/__init__.py
```

To train on a dataset with its default settings just pass the name of the dataset (all lowercase) :

```bash
python bin/train.py --dataset titan --model halfunet
```

You can override the dataset default configuration file:

```bash
python bin/train.py --dataset smeagol --model halfunet --dataset_conf config/smeagoldev.json
```

[Details on available datasets.](doc/features.md/#available-datasets)

### Training options

1. **Configuring the neural network**

To train on a dataset using a network with its default settings just pass the name of the architecture (all lowercase) as shown below:

```bash
python bin/train.py --dataset smeagol --model hilam

python bin/train.py --dataset smeagol --model halfunet
```

You can override some settings of the model using a json config file (here we increase the number of filter to 128 and use ghost modules):

```bash
python bin/train.py --dataset smeagol --model halfunet --model_conf config/halfunet128_ghost.json
```

[Details on available neural networks.](doc/features.md/#available-pytorchs-architecture)


2. **Changing the training strategy**

You can choose a training strategy using the **--strategy STRATEGY_NAME** cli argument:

```bash
python bin/train.py --dataset smeagol --model halfunet --strategy diff_ar
```

[Details on available training strategies.](doc/features.md/#available-training-strategies)

### Tracking experiment

We use [Tensorboad](https://www.tensorflow.org/tensorboard) to track the experiments. You can launch a tensorboard server using the following command:

**At Météo-France**:

**runai** will handle port forwarding for you.

```bash
runai tensorboard --logdir PATH_TO_YOUR_ROOT_PATH
```

**Elsewhere**

```bash
tensorboard --logdir PATH_TO_YOUR_ROOT_PATH
```

Then you can access the tensorboard server at the following address: `http://YOUR_SERVER_IP:YOUR_PORT/`


4. **Other training options**:

* `--seed SEED`           random seed (default: 42)
* `--loss LOSS`           Loss function to use (default: mse)
* `--lr LR`               learning rate (default: 0.001)
* `--val_interval VAL_INTERVAL`
                    Number of epochs training between each validation run (default: 1)
* `--epochs EPOCHS`       upper epoch limit (default: 200)
* `--profiler PROFILER`   Profiler required. Possibilities are ['simple', 'pytorch', 'None']
* `--batch_size BATCH_SIZE`
                    batch size
* `--precision PRECISION`
                    Numerical precision to use for model (32/16/bf16) (default: 32)
* `--limit_train_batches LIMIT_TRAIN_BATCHES`
                    Number of batches to use for training
* `--num_pred_steps_train NUM_PRED_STEPS_TRAIN`
                    Number of auto-regressive steps/prediction steps during training forward pass
* `--num_pred_steps_val_test NUM_PRED_STEPS_VAL_TEST`
                    Number of auto-regressive steps/prediction steps during validation and tests
* `--num_input_steps NUM_INPUT_STEPS`
                    Number of previous timesteps supplied as inputs to the model
* `--num_inter_steps NUM_INTER_STEPS`
                    Number of model steps between two samples
* `--no_log`
    When activated, log are not stored and models are not saved. Use in dev mode. (default: False)
* `--dev_mode`
    When activated, reduce number of epoch and steps. (default: False)
* `--load_model_ckpt LOAD_MODEL_CKPT`
    Path to load model parameters from (default: None)


You can find more details about all the `num_X_steps` options [here](doc/num_steps.md).

## Adding features and contributing

This [page](doc/add_features_contribute.md) explains how to:
* add a new neural network
* add a new dataset
* contribute to this project following our guidelines

## Design choices

The figure below illustrates the principal components of the Py4cast architecture.

![py4cast](doc/figs/py4cast_diag.jpg)

- We define **interface contracts** between the components of the system using [Python ABCs](https://docs.python.org/3/library/abc.html). As long as the Python classes respect the interface contract, they can be used interchangeably in the system and the underlying implementation can be very different. For instance datasets with any underlying storage (grib2, netcdf, mmap+numpy, ...) and real-time or ahead of time concat and pre-processing could be used with the same neural network architectures and training strategies. 

- **Adding a model, a dataset, a loss, a plot, a training strategy, ... should be as simple as creating a new Python class that complies with the interface contract**.

- Neural network architectures are Python classes that inherit from both **ModelABC** and PyTorch's **nn.Module**. The later means it is quick to insert a third-party pure PyTorch model in the system (see for instance the code for Lucidrains' Segformer or a U-Net).

- We use **dataclasses** and **dataclass_json** to define the settings whenever possible. This allows us to easily serialize and deserialize the settings to/from json files with Schema validation.

- The [NamedTensor](doc/features.md/#namedtensors) allows us to keep track of the physical/weather parameters along the features dimension and to pass a single consistent object in the system. It is also a way to factorize common operations on tensors (concat along features dimension, flatten in place, ...) while keeping the dimension and feature names metadata in sync.

- We use **PyTorch-lightning** to train the models. This allows us to easily scale the training to multiple GPUs and to use the same training loop for all the models. We also use the **PyTorch-lightning** logging system to log the training metrics and the hyperparameters.

### Ideas for future improvements

- Ideally, we could end up with a simple based class system for the training strategies to allow for easy addition of new strategies.

- The **ItemBatch** class attributes could be generalized to have multiple inputs, outputs and forcing tensors referenced by name, this would allow for more flexibility in the models and plug metnet-3 and Pangu.

- The distinction between **prognostic** and **diagnostic** variables should be made explicit in the system.

- We should probably reshape back the GNN outputs to (lat, lon) gridded shape as early as possible to have this as a common/standard output format for all the models. This would simplify the post-processing, plotting, ... We still have if statements in the code to handle the different output shapes of the models.
