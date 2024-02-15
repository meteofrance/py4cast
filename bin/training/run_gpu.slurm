#!/bin/bash
#
# ==============================================================================
#
# description     : Thin sbatch wrapper for running single node multiGPU training
# author          : Frank Guibert - DSM/LabIA
# company         : MF
# mail            : frank.guibert@meteo.fr
# date            : 2022-12-16
#
# ==============================================================================

#SBATCH --job-name=pnia-cha
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:v100:4
#SBATCH --time=0-23:00:00
#SBATCH --partition=node3

srun --nodes=1 --ntasks-per-node=1 --ntasks=1 --gres=gpu:v100:4 runai build_and_run_gpu $@
