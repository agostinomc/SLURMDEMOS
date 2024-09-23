#!/usr/bin/env bash
module load apptainer
#export APPTAINER_CACHEDIR=$WORK/.apptainer/cache

# define software stack that will become default soon (September 2024)
# see https://docs.hpc.gwdg.de/software/nhr_lmod/index.html
export PREFERRED_SOFTWARE_STACK=nhr-lmod
source /sw/etc/profile/profile.sh

rm ~/dlc-dlwgpu.sif
apptainer build --nv ~/dlc-dlwgpu.sif dlc-dlwgpu.def > build_dlc-dlwgpu.log 2>&1
