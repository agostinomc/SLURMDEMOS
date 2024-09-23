#!/bin/bash
#SBATCH --job-name=test-nn-gpu
#SBATCH -t 00:10:00                  # estimated time # TODO: adapt to your needs
#SBATCH -p kisski                    # the partition you are training on (i.e., which nodes), for nodes see sinfo -o "%25N  %5c  %10m  %32f  %10G %18P " | grep gpu
#SBATCH -G A100:1          # requesting GPU slices, see https://docs.hpc.gwdg.de/usage_guide/slurm/gpu_usage/index.html for more options
#SBATCH --nodes=1                    # total number of nodes
#SBATCH --ntasks=1                   # total number of tasks
#SBATCH --cpus-per-task 4            # number of CPU cores per task
#SBATCH --mail-type=all              # send mail when job begins and ends
#SBATCH --output=./slurm_files/slurm-%x-%j.out     # where to write output, %x give job name, %j names job id
#SBATCH --error=./slurm_files/slurm-%x-%j.err      # where to write slurm error

# load new software stack
# see https://docs.hpc.gwdg.de/software/software_stacks/index.html
export PREFERRED_SOFTWARE_STACK=nhr-lmod
source /sw/etc/profile/profile.sh

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
python -u test.py

# Run the script with logger:
#python -u train_with_logger.py -l ~/${SLURM_JOB_NAME}_${SLURM_JOB_ID}  -t True -p True -d True -s True -f True
