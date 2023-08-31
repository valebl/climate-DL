#!/bin/bash

#------------------#
# SLURM PARAMETERS #
#------------------#
TIME="24:00:00"
MEM="60G"
JOB_NAME="test_reg"
MAIL="vblasone@ictp.it"
LOG_PATH="/leonardo_work/ICT23_ESP_0/vblasone/climate-DL/local_single/reg_test/"

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
INPUT_PATH="/leonardo_work/ICT23_ESP_0/vblasone/climate-DL/data_fvg_preprocessed/"
OUTPUT_PATH="/leonardo_work/ICT23_ESP_0/vblasone/climate-DL/local_single/reg_test/"
INPUT_FILE="input_standard.pkl"
IDX_FILE="idx_train_reg.pkl"
LOG_FILE="log.txt"
TARGET_FILE="target_train_reg.pkl"
MASK_TARGET_FILE="mask_train_reg.pkl"
GRAPH_FILE="G_train.pkl"
SUBGRAPHS_FILE="subgraphs.pkl"
OUT_CHECKPOINT_FILE="checkpoint.pth"
OUT_LOSS_FILE="loss.csv"
WEIGHTS_FILE="weights_reg_train.pkl"

#---------------------#
# TRAINING PARAMETERS #
#---------------------#
PCT_TRAINING=1
EPOCHS=100
BATCH_SIZE=64
LR_STEP_SIZE=25
LR=0.0001
WEIGHT_DECAY=0.0
MODEL_NAME="Regressor_edges"
LOSS_FN="weighted_mse_loss"
MODEL_TYPE="reg"
CHECKPOINT_FILE="/leonardo_work/ICT23_ESP_0/vblasone/climate-DL/data_fvg_preprocessed/checkpoint_ae_e3.pth"
WANDB_PROJECT_NAME="Regressor-test-Leonardo"

#----------------------------------------#
# COORDINATES OF CONSIDERED SPATIAL AREA #
#----------------------------------------#
LON_MIN=12.75
LON_MAX=14.00
LAT_MIN=45.50
LAT_MAX=46.75

#-----------------#
# BOOLEAN OPTIONS #
#-----------------#
USE_ACCELERATE=true
LOAD_CHECKPOINT=true
CTD_TRAINING=false
FINE_TUNING=true


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

