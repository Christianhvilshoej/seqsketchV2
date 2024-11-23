#!/bin/sh
#BSUB -q gpuv100
#BSUB -J train_model
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 15:00
#BSUB -R "rusage[mem=8GB]"
#BSUB -u chr.hvilshoej@gmail.com
#BSUB -B
#BSUB -N
#BSUB -o train_%J.out
#BSUB -e train_%J.err

nvidia-smi

module load cuda/11.6
module load python3/3.10.14
unset PYTHONHOME
unset PYTHONPATH

source $HOME/miniconda3/bin/activate
conda activate ddpm

/appl/cuda/11.6.0/samples/bin/x86_64/linux/release/deviceQuery

python train.py --log_interval=5000
