# PY4CAST

![Unit Tests](https://github.com/meteofrance/py4cast/actions/workflows/tests.yml/badge.svg)

This project, built using **PyTorch** and **PyTorch-lightning**, is designed to train a variety of Neural Network architectures (GNNs, CNNs, Vision Transformers, ...) on various weather forecasting datasets. This is a **Work in Progress**, intended to share ideas and design concepts with partners.

Developped at Météo-France by **DSM/AI Lab** and **CNRM/GMAP/PREV**.

Contributions are welcome (Issues, Pull Requests, ...).

This project is licensed under the [APACHE 2.0 license.](LICENSE-2.0.txt)

![Forecast humidity](doc/figs/2023061812_aro_r2_2m_crop.gif)
![Forecast precip](doc/figs/2023061812_aro_tp_0m_crop.gif)

# Acknowledgements

This project started as a fork of neural-lam, a project by Joel Oskarsson, see [here](https://github.com/mllam/neural-lam). Many thanks to Joel for his work!


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
    7. [Inference](#inference)
    8. [Making animated plots comparing multiple models](#making-animated-plots-comparing-multiple-models)
4. [Contributing new features](#adding-features-and-contributing)
    1. [Adding a neural network architecture](doc/add_features_contribute.md#adding-a-new-neural-network-architecture-to-the-project)
    2. [Adding a dataset](doc/add_features_contribute.md#adding-a-new-dataset)
    3. [Adding plots](doc/add_features_contribute.md#adding-training-plots)
5. [Design choices](#design-choices)
6. [Unit tests](doc/add_features_contribute.md#unit-tests)
7. [Continuous Integration](doc/add_features_contribute.md#continuous-integration)


## Overview

* 7 neural network architectures : Half-Unet, U-Net, SegFormer, SwinUnetR, HiLam, GraphLam, UnetR++
* 1 dataset with samples available on Huggingface : Titan
* 2 training strategies : Scaled Auto-regressive steps, Differential Auto-regressive steps
* 4 losses: Scaled RMSE, Scaled L1, Weighted MSE, Weighted L1
* neural networks as simple torch.nn.Module
* training with pytorchlightning
* simple interfaces to easily add a new dataset, neural network, training strategy or loss
* simple command line to lauch a training
* config files to change the parameters of your dataset or neural network during training
* experiment tracking with tensorboard and plots of forecasts with matplotlib
* implementation of [NamedTensors](doc/features.md#namedtensors) to tracks features and dimensions of tensors at each step of the training

See [here](doc/features.md) for details on the available datasets, neural networks, training strategies, losses, and explanation of our NamedTensor.

## Installation

Start by cloning the repository:
```bash
git clone https://github.com/meteofrance/py4cast.git
cd py4cast
```

### Setting environment variables

In order to be able to run the code on different machines, some environment variables can be set.
You may add them in your `.bashrc` or modify them just before launching an experiment.

- `PY4CAST_ROOTDIR` : Specify the ROOT DIR for your experiment. It also modifies the CACHE_DIR. This is where the files created during the experiment will be stored.
- `PY4CAST_SMEAGOL_PATH`: Specify where the smeagol dataset is stored. Only needed if you want to use the smeagol dataset.
- `PY4CAST_TITAN_PATH`: Specify where the titan dataset is stored. Only needed if you want to use the titan dataset.

This should be done by
```sh
export PY4CAST_ROOTDIR="/my/dir/"
```

You **MUST** export **PY4CAST_ROOTDIR** to make py4cast work, you can use for instance the existing **SCRATCH** env var:
```bash
export PY4CAST_ROOTDIR=$SCRATCH/py4cast
```

If **PY4CAST_ROOTDIR** is not exported py4cast will default to use **/scratch/shared/py4cast** as its root directory, leading to Exceptions if this directory does not exist or if it is not writable.

### At Météo-France

When working at Météo-France, you can use either runai + Docker or Conda/Micromamba to setup a working environment. On the AI Lab cluster we recommend using runai, Conda on our HPC.

See the [runai repository](https://git.meteo.fr/dsm-labia/monorepo4ai) for installation instructions.

For HPC, see the related doc (doc/install/install_MF.md) to get the right installation settings.

### Install with conda

You can install a conda environment, including `py4cast` in editable mode, using
```sh
conda env create --file env.yaml
```

From an exixting conda environment, you can now install manually `py4cast` in development mode using
```sh
conda install conda-build -n py4cast
conda develop .
```
or
```sh
pip install --editable .
```

In case the install fail because some dependencies are not found or are in conflict, please look at the [installation known issues](doc/known_issues.md#installation).


### Install with micromamba

Please install the environment using :
```sh
micromamba create -f env.yaml
```

From an exixting micromamba environment, you can now install manually `py4cast` in editable mode using
```sh
pip install --editable .
```

### Build docker image

To build the docker image please use the `oci-image-build.sh` script.
For Meteo-France user, you should export the variable `INJECT_MF_CERT` to use the Meteo-France certificate
```sh
export INJECT_MF_CERT=1
```
Then, build with the following command 
```sh
bash ./oci-image-build.sh --runtime docker
```
By default, the `CUDA` and `pytorch` version are extracted from the `env.yaml` reference file. Nevertheless, for test purpose, you can set the **PY4CAST_CUDA_VERSION** and **PY4CAST_TORCH_VERSION** to override the default versions.

### Build podman image

As an alternative to docker, you can use podman to build the image.

<details>
<summary>Click to expand</summary>

To build the podman image please use the `oci-image-build.sh` script.
```sh
bash ./oci-image-build.sh --runtime podman
```
By default, the `CUDA` and `pytorch` version are extracted from the `env.yaml` reference file. Nevertheless, for test purpose, you can set the **PY4CAST_CUDA_VERSION** and **PY4CAST_TORCH_VERSION** to override the default versions.

</details>

### Convert to Singularity image

From a previously built docker or podman image, you can convert it to the singularity format.

<details>
<summary>Click to expand</summary>

To convert the previously built image to a Singularity container, you have to first save the image as a `tar` file:
```sh
docker save py4cast:your_tag -o py4cast-your_tag.tar
```
or with podman:
```sh
podman save --format oci-archive py4cast:your_tag -o py4cast-your_tag.tar
```

Then, build the singularity image with:
```sh
singularity build py4cast-your_tag.sif docker-archive://py4cast-your_tag.tar
```
Please, be sure to get enough free disk space to store the .tar and .sif files.

</details>

## Usage

### Docker

From your `py4cast` source directory, to run an experiment using the docker image you need to mount in the container :
- The dataset path
- The py4cast sources
- The PY4CAST_ROOTDIR path

Here is an example of command to run a "dev_mode" training of the HiLam model with the TITAN dataset, using all the GPUs:
```sh
docker run \
    --name hilam-titan \
    --rm \
    --gpus all \
    -v ./${HOME} \
    -v <path-to-datasets>/TITAN:/dataset/TITAN \
    -v <your_py4cast_root_dir>:<your_py4cast_root_dir> \
    -e PY4CAST_ROOTDIR=<your_py4cast_root_dir> \
    -e PY4CAST_TITAN_PATH=/dataset/TITAN \
    py4cast:<your_tag> \
    bash -c " \
        pip install -e . &&  \
        python bin/pyr4cast.py
    "
```

### Podman

<details>
<summary>Click to expand</summary>

From your `py4cast` source directory, to run an experiment using the podman image you need to mount in the container :
- The dataset path
- The py4cast sources
- The PY4CAST_ROOTDIR path

Here is an example of command to run a "dev_mode" training of the HiLam model with the TITAN dataset, using all the GPUs:
```sh
podman run \
    --name hilam-titan \
    --rm \
    --device nvidia.com/gpu=all \
    --ipc=host \
    --network=host \
    -v ./${HOME} \
    -v <path-to-datasets>/TITAN:/dataset/TITAN \
    -v <your_py4cast_root_dir>:<your_py4cast_root_dir> \
    -e PY4CAST_ROOTDIR=<your_py4cast_root_dir> \
    -e PY4CAST_TITAN_PATH=/dataset/TITAN \
    py4cast:<your_tag> \
    bash -c " \
        pip install -e . &&  \
        python bin/py4cast.py
    "
```
</details>

### Singularity

<details>
<summary>Click to expand</summary>

From your `py4cast` source directory, to run an experiment using a singularity container you need to mount in the container :
- The dataset path
- The PY4CAST_ROOTDIR path

Here is an example of command to run a "dev_mode" training of the HiLam model with the TITAN dataset:
```sh
PY4CAST_TITAN_PATH=/dataset/TITAN \
PY4CAST_ROOTDIR=<your_py4cast_root_dir> \
singularity exec \
    --nv \
    --bind <path-to-datasets>/TITAN:/dataset/TITAN \
    --bind <your_py4cast_root_dir>:<your_py4cast_root_dir> \
    py4cast-<your_tag>.sif \
    bash -c " \
        pip install -e . &&  \
        python bin/py4cast.py
    "
```
</details>

### runai

For now this works only for internal Météo-France users.

<details>
<summary>Click to expand</summary>

`runai` commands must be issued at the root directory of the `py4cast` project:

1. Run an interactive training session

```bash
runai gpu_play 4
runai build
runai exec_gpu python bin/py4cast.py
```

2. Train using sbatch single node multi-GPUs

```bash
export RUNAI_GRES="gpu:v100:4"
runai sbatch python bin/py4cast.py
```

3. Train using sbatch multi nodes multi GPUs

Here we use 2 nodes with 4 GPUs each.

```bash
export RUNAI_SLURM_NNODES=2
export RUNAI_GRES="gpu:v100:4"
runai sbatch_multi_node python bin/py4cast.py
```

For the rest of the documentation, you must preprend each python command with `runai exec_gpu`.
</details>


### Conda or Micromamba

Once your micromamba environment is setup, you should :
 - activate your environment `conda activate py4cast` or `micromamba activate nlam`
 - launch a training

A very simple training can be launch (on your current node)
```sh
python bin/py4cast.py
```

#### Example of script  to launch on gpu

To do so, you will need to create a small `sh` script.

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
cd $PY4CAST_PATH # Go to Py4CAST (you can either add an environment variable or hard code it here).
# Launch your favorite command.
srun python bin/py4cast.py
```


Then just launch this script using

```sh
sbatch my_tiny_script.sh
```
**NB** Note that you may have some trouble with SSL certificates (for cartopy). You may need to explicitely export the certificate as :
```sh
 export SSL_CERT_FILE="/opt/softs/certificats/proxy1.pem"
```
with the proxy path depending on your machine.

### Dataset initialization

As in neural-lam, before training you must first compute the mean and std of each feature.

To compute the stats of the Titan dataset:

```bash
python py4cast/datasets/titan/__init__.py
```

### How to configure arguments with config.yaml

The LightningCLI use a config.yaml file to parse its arguments. Any argument can be specified in this file, provided that it is also an argument of the Trainer (specific to the CLI), the LightningModule, or the DataModule.
- Any arg associated with the Trainer will be refered as trainer.arg
- Any arg associated with the LightningModule will be refered as model.arg
- Any arg associated with the DataModule will be refered as data.arg
Config file path : "py4cast/config/config_cli_autoregressive.yaml"

#### Dataset config 

You can override the dataset default configuration file:

```bash
data: 
  dataset_name: "titan" # Replace with actual dataset name
  dataset_conf: "config/datasets/titan_full.json" # Replace with actual config path
  config_override: 
```

[Details on available datasets.](doc/features.md/#available-datasets)

#### Model config

You can override the model default configuration file (here we increase the number of filter to 128 and use ghost modules):

```bash
model:
  model_conf: config/halfunet128_ghost.json # Replace with actual config path
  model_name: "halfunet" # Replace with actual model name
```

[Details on available neural networks.](doc/features.md/#available-pytorchs-architecture)


3. **Other training options**:

```bash
[-h] [-c CONFIG] [--print_config[=flags]] [--seed_everything SEED_EVERYTHING] [--trainer CONFIG]
                      [--trainer.accelerator.help CLASS_PATH_OR_NAME] [--trainer.accelerator ACCELERATOR] [--trainer.strategy.help CLASS_PATH_OR_NAME]
                      [--trainer.strategy STRATEGY] [--trainer.devices DEVICES] [--trainer.num_nodes NUM_NODES] [--trainer.precision PRECISION]
                      [--trainer.logger.help CLASS_PATH_OR_NAME] [--trainer.logger LOGGER] [--trainer.callbacks.help CLASS_PATH_OR_NAME]
                      [--trainer.callbacks CALLBACKS] [--trainer.fast_dev_run FAST_DEV_RUN] [--trainer.max_epochs MAX_EPOCHS]
                      [--trainer.min_epochs MIN_EPOCHS] [--trainer.max_steps MAX_STEPS] [--trainer.min_steps MIN_STEPS] [--trainer.max_time MAX_TIME]
                      [--trainer.limit_train_batches LIMIT_TRAIN_BATCHES] [--trainer.limit_val_batches LIMIT_VAL_BATCHES]
                      [--trainer.limit_test_batches LIMIT_TEST_BATCHES] [--trainer.limit_predict_batches LIMIT_PREDICT_BATCHES]
                      [--trainer.overfit_batches OVERFIT_BATCHES] [--trainer.val_check_interval VAL_CHECK_INTERVAL]
                      [--trainer.check_val_every_n_epoch CHECK_VAL_EVERY_N_EPOCH] [--trainer.num_sanity_val_steps NUM_SANITY_VAL_STEPS]
                      [--trainer.log_every_n_steps LOG_EVERY_N_STEPS] [--trainer.enable_checkpointing {true,false,null}]
                      [--trainer.enable_progress_bar {true,false,null}] [--trainer.enable_model_summary {true,false,null}]
                      [--trainer.accumulate_grad_batches ACCUMULATE_GRAD_BATCHES] [--trainer.gradient_clip_val GRADIENT_CLIP_VAL]
                      [--trainer.gradient_clip_algorithm GRADIENT_CLIP_ALGORITHM] [--trainer.deterministic DETERMINISTIC]
                      [--trainer.benchmark {true,false,null}] [--trainer.inference_mode {true,false}] [--trainer.use_distributed_sampler {true,false}]
                      [--trainer.profiler.help CLASS_PATH_OR_NAME] [--trainer.profiler PROFILER] [--trainer.detect_anomaly {true,false}]
                      [--trainer.barebones {true,false}] [--trainer.plugins.help CLASS_PATH_OR_NAME] [--trainer.plugins PLUGINS]
                      [--trainer.sync_batchnorm {true,false}] [--trainer.reload_dataloaders_every_n_epochs RELOAD_DATALOADERS_EVERY_N_EPOCHS]
                      [--trainer.default_root_dir DEFAULT_ROOT_DIR] [--model CONFIG] --model.batch_size BATCH_SIZE [--model.model_conf MODEL_CONF]
                      [--model.model_name MODEL_NAME] [--model.lr LR] [--model.loss_name LOSS_NAME] [--model.num_input_steps NUM_INPUT_STEPS]
                      [--model.num_pred_steps_train NUM_PRED_STEPS_TRAIN] [--model.num_pred_steps_val_test NUM_PRED_STEPS_VAL_TEST]
                      [--model.num_samples_to_plot NUM_SAMPLES_TO_PLOT] [--model.training_strategy TRAINING_STRATEGY]
                      [--model.len_train_loader LEN_TRAIN_LOADER] [--model.save_path SAVE_PATH] [--model.use_lr_scheduler {true,false}]
                      [--model.precision PRECISION] [--model.no_log {true,false}] [--model.channels_last {true,false}]
                      [--model.save_weights_path SAVE_WEIGHTS_PATH] [--data CONFIG] --data.dataset_name DATASET_NAME
                      [--data.dataset_conf DATASET_CONF] [--data.config_override CONFIG_OVERRIDE] [--data.num_input_steps NUM_INPUT_STEPS]
                      [--data.num_pred_steps_train NUM_PRED_STEPS_TRAIN] [--data.num_pred_steps_val_test NUM_PRED_STEPS_VAL_TEST]
                      [--optimizer.help CLASS_PATH_OR_NAME] [--optimizer CONFIG | CLASS_PATH_OR_NAME | .INIT_ARG_NAME VALUE]
                      [--lr_scheduler.help CLASS_PATH_OR_NAME] [--lr_scheduler CONFIG | CLASS_PATH_OR_NAME | .INIT_ARG_NAME VALUE]
```

You can find more details about all the `num_X_steps` options [here](doc/num_steps.md).


### Tracking experiment

#### Tensorboard

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


#### MLFlow

Optionally, you can use MLFlow, in addition to Tensorboard, to track your experiment and log your model. To activate the MLFlow logger simply add the `--mlflow_log` option on the `bin/train.py` command line.

**Local usage**

Without a MLFlow server, the logs are stored in your root path, at `PY4CAST_ROOTDIR/logs/mlflow`.

**With a MLFlow server**

If you have a MLFow server you can configure your training environment to push the logs on the remote server. A set of [environment variables](https://mlflow.org/docs/latest/cli.html#mlflow-server) are available to do that.

For exemple, you can export the following variable in your training environment:

```bash
export MLFLOW_TRACKING_URI=https://my.mlflow.server.com/
export MLFLOW_TRACKING_USERNAME=<your-mlflow-user>
export MLFLOW_TRACKING_PASSWORD=<your-mlflow-pwd>
export MLFLOW_EXPERIMENT_NAME=py4cast/unetrpp
```

### Inference

Inference is done by running the `bin/inference.py` script. This script will load a model and run it on a dataset using the training parameters (dataset config, timestep options, ...).

```bash
usage: python bin/inference.py [-h] [--model_path MODEL_PATH] [--dataset DATASET] [--infer_steps INFER_STEPS] [--date DATE]

options:
  -h, --help            show this help message and exit
  --model_path MODEL_PATH
                        Path to the model checkpoint
  --date DATE
                        Date of the sample to infer on. Format:YYYYMMDDHH
  --dataset DATASET
                        Name of the dataset to use (typically the same as has been used for training)
  --dataset_conf DATASET_CONF
                        Name of the dataset config file (json, to change e.g dates, leadtimes, etc)
  --infer_steps INFER_STEPS
                        Number of auto-regressive steps/prediction steps during the inference
   --precision PRECISION
                        floating point precision for the inference (default: 32)
   --grib BOOL
                        Whether the outputs should be saved as grib, needs saving conf.
   --saving_conf SAVING_CONF
                        Name of the config file for write settings (json)
```

A simple example of inference is shown below:

```bash
 runai exec_gpu python bin/inference.py --model_path /scratch/shared/py4cast/logs/camp0/poesy/halfunet/sezn_run_dev_12 --date 2021061621 --dataset poesy_infer --infer_steps 2
```

### Making animated plots comparing multiple models

You can compare multiple trained models on specific case studies and visualize the forecasts on animated plots with the `bin/gif_comparison.py`. See example of GIF at the beginning of the README.

Warnings:
- For now this script only works with models trained with Titan dataset.
- If you want to use AROME as a model, you have to manually download the forecast before.

```bash
Usage: gif_comparison.py [-h] --ckpt CKPT --date DATE [--num_pred_steps NUM_PRED_STEPS]

options:
  -h, --help            show this help message and exit
  --ckpt CKPT           Paths to the model checkpoint or AROME
  --date DATE           Date for inference. Format YYYYMMDDHH.
  --num_pred_steps NUM_PRED_STEPS
                        Number of auto-regressive steps/prediction steps.

example: python bin/gif_comparison.py --ckpt AROME --ckpt /.../logs/my_run/epoch=247.ckpt
                                      --date 2023061812 --num_pred_steps 10
```

### Scoring and comparing models

The `bin/test.py` script will compute and save metrics on the validation set, on as many auto-regressive prediction steps as you want.

```bash
python bin/test.py PATH_TO_CHECKPOINT --num_pred_steps 24
```

Once you have executed the `test.py` script on all the models you want, you can compare them with `bin/scores_comparison.py`:

```bash
python bin/scores_comparison.py --ckpt PATH_TO_CKPT_0  --ckpt PATH_TO_CKPT_1
```

**Warning**: For now `bin/scores_comparison.py` only works with models trained with Titan dataset

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

- Dataset produce **Item**, collated into **ItemBatch**, both having **NamedTensor** attributes.

- Dataset produce tensors with the following dimensions: (batch, timestep, lat, lon, features). Models can flatten or reshape spatial dimension in the **prepare_batch** but the rest of the system expects **features** to be **always the last dimension of the tensors**.

- Neural network architectures are Python classes that inherit from both **ModelABC** and PyTorch's **nn.Module**. The later means it is quick to insert a third-party pure PyTorch model in the system (see for instance the code for Lucidrains' Segformer or a U-Net).

- We use **dataclasses** and **dataclass_json** to define the settings whenever possible. This allows us to easily serialize and deserialize the settings to/from json files with Schema validation.

- The [NamedTensor](doc/features.md/#namedtensors) allows us to keep track of the physical/weather parameters along the features dimension and to pass a single consistent object in the system. It is also a way to factorize common operations on tensors (concat along features dimension, flatten in place, ...) while keeping the dimension and feature names metadata in sync.

- We use **PyTorch-lightning** to train the models. This allows us to easily scale the training to multiple GPUs and to use the same training loop for all the models. We also use the **PyTorch-lightning** logging system to log the training metrics and the hyperparameters.

### Ideas for future improvements

- Ideally, we could end up with a simple based class system for the training strategies to allow for easy addition of new strategies.

- The **ItemBatch** class attributes could be generalized to have multiple inputs, outputs and forcing tensors referenced by name, this would allow for more flexibility in the models and plug metnet-3 and Pangu.

- The distinction between **prognostic** and **diagnostic** variables should be made explicit in the system.

- We should probably reshape back the GNN outputs to (lat, lon) gridded shape as early as possible to have this as a common/standard output format for all the models. This would simplify the post-processing, plotting, ... We still have if statements in the code to handle the different output shapes of the models.
