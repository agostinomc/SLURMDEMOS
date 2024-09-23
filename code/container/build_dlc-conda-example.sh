#!/usr/bin/env bash
module load apptainer

#export APPTAINER_CACHEDIR=$WORK/.apptainer/cache


export PREFERRED_SOFTWARE_STACK=nhr-lmod
source /sw/etc/profile/profile.sh

rm ~/dlc-conda-example.sif
apptainer build --nv ~/dlc-conda-example.sif dlc-conda-example.def > build_dlc-conda-example.log 2>&1
