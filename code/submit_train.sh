#!/bin/bash
#SBATCH --job-name=train-nn-gpu
#SBATCH -t 00:20:00                  # estimated time # TODO: adapt to your needs
#SBATCH --nodes=1                    # total number of nodes
#SBATCH --ntasks=1                   # total number of tasks
#SBATCH --cpus-per-task 4            # number of CPU cores per task
#SBATCH --output=./slurm_files/slurm-%x-%j.out     # where to write output, %x give job name, %j names job id
#SBATCH --error=./slurm_files/slurm-%x-%j.err      # where to write slurm error


module load miniconda3
module load cuda
source activate dl-gpu # Or whatever you called your environment.

# Printing out some info.
echo "Submitting job with sbatch from directory: ${SLURM_SUBMIT_DIR}"
echo "Home directory: ${HOME}"
echo "Working directory: $PWD"
echo "Current node: ${SLURM_NODELIST}"

# For debugging purposes.
python --version
python -m torch.utils.collect_env
nvcc -V

# Run the script:
python -u train.py

# Run the script with logger:
#python -u train_with_logger.py -l ~/${SLURM_JOB_NAME}_${SLURM_JOB_ID}  -t True -p True -d True -s True -f True
