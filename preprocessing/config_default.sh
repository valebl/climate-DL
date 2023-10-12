#!/bin/bash

#------------------#
# SLURM PARAMETERS #
#------------------#
TIME="24:00:00"
MEM="0"
JOB_NAME="preprocessing"
MAIL="vblasone@ictp.it"
LOG_PATH="/leonardo_work/ICT23_ESP_0/SHARED/preprocessed/fvg/"

#-------------------------------#
# CHOOSE PREP PHASES TO PERFORM #
#-------------------------------#
PERFORM_PHASE_1A=false
PERFORM_PHASE_1B=false
PERFORM_PHASE_2=true

#---------------------------#
# INPUT DATA TO PERSONALIZE #
#---------------------------#
SOURCE_PATH="/leonardo/home/userexternal/vblasone/.bashrc"
INPUT_PATH_PHASE_1A="/leonardo_work/ICT23_ESP_0/SHARED/ERA5/"
INPUT_PATH_PHASE_1B="/leonardo_work/ICT23_ESP_0/SHARED/preprocessed/fvg/"
INPUT_PATH_PHASE_2="/leonardo_work/ICT23_ESP_0/SHARED/"
OUTPUT_PATH="/leonardo_work/ICT23_ESP_0/SHARED/preprocessed/fvg/"
ENV_PATH="/leonardo_work/ICT23_ESP_0/env/GNNenv"
LOG_FILE="log.txt"

PREFIX_PHASE_1="sliced_"

LOAD_STATS=true
STATS_PATH="/leonardo_work/ICT23_ESP_0/SHARED/preprocessed/italy/"
MEANS_FILE="means.pkl"
STDS_FILE="stds.pkl"

SUFFIX_PHASE_2=""
PRECOMPUTED_STATS_FILE="TOPO/z_stats_italy.pkl"

INTERVAL=0.25
TIME_DIM=140256
VARIABLES_LIST="q t u v z"

## fvg
LON_MIN=12.75
LON_MAX=14.00
LAT_MIN=45.50
LAT_MAX=46.75


#--------------do not change here below------------------

PHASE_1A_PATH="/leonardo_work/ICT23_ESP_0/SHARED/climate-DL/preprocessing/slice_and_merge_era5.sh"
PHASE_1B_PATH="/leonardo_work/ICT23_ESP_0/SHARED/climate-DL/preprocessing/preprocessing_era5.py"
PHASE_2_PATH="/leonardo_work/ICT23_ESP_0/SHARED/climate-DL/preprocessing/preprocessing_graphs_and_targets.py"

if [ ${LOAD_STATS} = true ] ; then
	LOAD_STATS="--load_stats"
else
	LOAD_STATS="--no-load_stats"
fi

