#!/bin/bash
#SBATCH --job-name=train-nn-gpu-dlc
#SBATCH -t 00:20:00                  # estimated time # TODO: adapt to your needs
#SBATCH -p kisski              # the partition you are training on (i.e., which nodes), for nodes see sinfo -o "%25N  %5c  %10m  %32f  %10G %18P " | grep gpu
#SBATCH -G A100:1                   # take 1 GPU, see https://www.hlrn.de/doc/display/PUB/GPU+Usage for more options
#SBATCH --mem-per-gpu=5G             # setting the right constraints for the splitted gpu partitions
#SBATCH --nodes=1                    # total number of nodes
#SBATCH --ntasks=1                   # total number of tasks
#SBATCH --cpus-per-task=4            # number cores per task
#SBATCH --output=./slurm_files/slurm-%x-%j.out     # where to write output, %x give job name, %j names job id
#SBATCH --error=./slurm_files/slurm-%x-%j.err      # where to write slurm error

module load apptainer

# Printing out some info.
echo "Submitting job with sbatch from directory: ${SLURM_SUBMIT_DIR}"
echo "Home directory: ${HOME}"
echo "Working directory: $PWD"
echo "Current node: ${SLURM_NODELIST}"

# For debugging purposes.
echo ""
echo "test environment"
echo "-----------------------------------------------------------------------"
apptainer exec --nv --bind /scratch /path/to/dlc-dlwgpu.sif python /path/to/deep-learning-with-gpu-cores/code/test_env.py #TODO adapt path
echo "-----------------------------------------------------------------------"

# Run the script:

echo ""
echo "model training"
echo "-----------------------------------------------------------------------"
apptainer exec --nv --bind /scratch /path/to/dlc-dlwgpu.sif python -u /path/to/deep-learning-with-gpu-cores/code/train.py #TODO adapt path
echo "-----------------------------------------------------------------------"
