# Installation on Idris Jean-Zay supercomputer

## User setup

Do not use your HOME folder for anything, it doesn't have any space available. Use the WORK space for your code and python packages:

```bash
mkdir $WORK/.local
mkdir $WORK/.conda
mkdir $WORK/.cache
ln -s $WORK/.local $HOME
ln -s $WORK/.conda $HOME
ln -s $WORK/.cache $HOME
```

## Project setup

Install conda env:

```bash
cd $WORK
git clone https://github.com/USERNAME/py4cast/tree/main

module purge
module load git
module load python/3.10.4

conda create -n py4cast python=3.10 -c conda-forge
conda activate py4cast

pip install -r requirements.txt
pip install pyg-lib==0.4.0 torch-scatter==2.1.2 torch-sparse==0.6.18 torch-cluster==1.6.2 torch-geometric==2.3.1 -f https://data.pyg.org/whl/torch-2.1.2+cpu.html
```

Set env vars:
```bash
export PYTHONPATH=.
export PY4CAST_ROOTDIR=.
export PY4CAST_TITAN_PATH=/lustre/fsn1/projects/rech/dxp/commun/Titan/
```

## Launch jobs

* Data preprocessing interactive session on CPU node: `srun --pty --nodes=1 --time=2:00:00 --account=dxp@v100 -C v100 --partition=prepost bash`

* Interactive session on dev A100 node: `srun --pty --nodes=1 --ntasks-per-node=8 --cpus-per-task=8 --hint=nomultithread --gres=gpu:8 --qos qos_gpu_a100-dev --account=dxp@a100 -C a100 --time=2:00:00 bash`

* Job on one A100 octo-gpu node: `sbatch PATH_TO_SBATCH_CARD.sh`

and write your SBATCH card (don't forget to change the paths and execution time):

```bash
#!/bin/bash
#!/usr/bin/bash

#SBATCH --job-name=train_py4cast
#SBATCH --nodes=1
#SBATCH --output=/lustre/fswork/projects/rech/dxp/USERNAME/py4cast/slurm_A100_8GPU.%j.out
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:8
#SBATCH --hint=nomultithread
#SBATCH --time=01:50:00
#SBATCH --account=dxp@a100
#SBATCH -C a100

echo `date`
module purge
module load git
module load python/3.10.4
conda activate py4cast

set -x

srun python bin/train.py --dataset titan --dataset_conf config/datasets/titan_full.json --model unetrpp --model_conf config/models/unetrpp161024_linear_up.json --epochs 10 --batch_size 8 --num_workers 8 --num_pred_steps_val_test 1 --num_input_steps 1 --strategy scaled_ar --prefetch_factor 2 --seed 42
```