#!/bin/bash

#------------------#
# SLURM PARAMETERS #
#------------------#
TIME="24:00:00"
MEM="0"
JOB_NAME="test_reg_ft"
MAIL="vblasone@ictp.it"
LOG_PATH="/leonardo_work/ICT23_ESP_0/vblasone/climate-DL/local_single/test_reg_large_ft/"

#--------------------------------------------#
# PATHS (conda env, main, accelerate config) #
#--------------------------------------------#
SOURCE_PATH="/leonardo/home/userexternal/vblasone/.bashrc"
ENV_PATH="/leonardo_work/ICT23_ESP_0/env/GNNenv"
MAIN_PATH="/leonardo_work/ICT23_ESP_0/vblasone/climate-DL/local_single/"
ACCELERATE_CONFIG_PATH="/leonardo/home/userexternal/vblasone/.cache/huggingface/accelerate/default_config_4.yaml"

#-------------------------------#
# INPUT/OUTPUT FILES PARAMETERS #
#-------------------------------#
INPUT_PATH="/leonardo_work/ICT23_ESP_0/SHARED/preprocessed/north_italy/"
OUTPUT_PATH="/leonardo_work/ICT23_ESP_0/vblasone/climate-DL/local_single/test_reg_large_ft/"
INPUT_FILE="input_standard.pkl"
IDX_FILE="idx_train_reg.pkl"
LOG_FILE="log.txt"
TARGET_FILE="target_train_reg.pkl"
MASK_TARGET_FILE="mask_train_reg.pkl"
GRAPH_FILE="G_large.pkl"
SUBGRAPHS_FILE="subgraphs_large.pkl"
OUT_CHECKPOINT_FILE="checkpoint.pth"
OUT_LOSS_FILE="loss.csv"
WEIGHTS_FILE="weights_reg_train.pkl"
IDX_TIME_FILE="idx_time_train.pkl"

#---------------------#
# TRAINING PARAMETERS #
#---------------------#
PCT_TRAINING=1
EPOCHS=100
BATCH_SIZE=10
LR_STEP_SIZE=25
LR=0.0001
WEIGHT_DECAY=0.0
MODEL_NAME="Regressor_edges_large"
LOSS_FN="weighted_mse_loss"
MODEL_TYPE="reg"
CHECKPOINT_FILE="/leonardo_work/ICT23_ESP_0/vblasone/climate-DL/local_single/reg_north_italy/checkpoint_29.pth"
#"/leonardo_work/ICT23_ESP_0/vblasone/climate-DL/data_fvg_preprocessed/checkpoint_ae_e3.pth"
WANDB_PROJECT_NAME="Test-reg-large-ft"

#----------------------------------------#
# COORDINATES OF CONSIDERED SPATIAL AREA #
#----------------------------------------#
LON_MIN=6.75
LON_MAX=14.00
LAT_MIN=43.75
LAT_MAX=47.00

#-----------------#
# BOOLEAN OPTIONS #
#-----------------#
USE_ACCELERATE=true
LOAD_CHECKPOINT=true
CTD_TRAINING=true
FINE_TUNING=true
LARGE_GRAPH=true

#--------------do not change here below------------------


if [ ${USE_ACCELERATE} = true ] ; then
	USE_ACCELERATE="--use_accelerate"
else
	USE_ACCELERATE="--no-use_accelerate"
fi

if [ ${LOAD_CHECKPOINT} = true ] ; then
	LOAD_CHECKPOINT="--load_checkpoint"
else
	LOAD_CHECKPOINT="--no-load_checkpoint"
fi

if [ ${CTD_TRAINING} = true ] ; then
	CTD_TRAINING="--ctd_training"
else
	CTD_TRAINING="--no-ctd_training"
fi

if [ ${FINE_TUNING} = true ] ; then
	FINE_TUNING="--fine_tuning"
else
	FINE_TUNING="--no-fine_tuning"
fi

if [ ${LARGE_GRAPH} = true ] ; then
	LARGE_GRAPH="--large_graph"
else
	LARGE_GRAPH="--no-large_graph"
fi

if [ -z ${WEIGHTS_FILE} ]; then
	WEIGHTS=""
else
	WEIGHTS="--weights_file=${WEIGHTS_FILE}"
fi

