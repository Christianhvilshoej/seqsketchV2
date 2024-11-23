# here follow the commands you want to execute with input.in as the input file
### General options
#BSUB -q gpuv100
#BSUB -J train_single
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 24:00
#BSUB -R "rusage[mem=16GB]"
#BSUB -u chr.hvilshoej@gmail.com
#BSUB -B
#BSUB -N
#BSUB -o train_single.out
#BSUB -e train_single.err
# -- end of LSF options --

nvidia-smi
# Load the cuda module
module load cuda/11.6

#source cs492d/bin/activate
source $HOME/miniconda3/bin/activate
conda activate seqsketch

train --config baseline_single.yaml
