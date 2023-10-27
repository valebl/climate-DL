#!/bin/bash

#------------------#
# SLURM PARAMETERS #
#------------------#
TIME="24:00:00"
MEM="60G"
JOB_NAME="pred"
MAIL="vblasone@ictp.it"
LOG_PATH="/leonardo_work/ICT23_ESP_0/SHARED/predictions/fvg/"

#--------------------------------------------#
# PATHS (conda env, main, accelerate config) #
#--------------------------------------------#
SOURCE_PATH="/leonardo/home/userexternal/vblasone/.bashrc"
ENV_PATH="/leonardo_work/ICT23_ESP_0/env/GNNenv"
MAIN_PATH="/leonardo_work/ICT23_ESP_0/SHARED/climate-DL/predictions/"

#-------------------------------#
# INPUT/OUTPUT FILES PARAMETERS #
#-------------------------------#
INPUT_PATH="/leonardo_work/ICT23_ESP_0/SHARED/preprocessed/fvg/"
OUTPUT_PATH="/leonardo_work/ICT23_ESP_0/SHARED/predictions/fvg/"
INPUT_FILE="input_standard.pkl"
TEST_GRAPH_FILE="G_test.pkl"
LOG_FILE="log.txt"
SUBGRAPHS_FILE="subgraphs.pkl"
CHECKPOINT_CL="/leonardo_work/ICT23_ESP_0/vblasone/climate-DL/local_single/test_cl_ann/checkpoint_5.pth"
CHECKPOINT_REG="/leonardo_work/ICT23_ESP_0/vblasone/climate-DL/local_single/reg_north_italy/checkpoint_29.pth"
MODEL_NAME_CL="Classifier_edges_test"
MODEL_NAME_REG="Regressor_edges_test"
OUTPUT_FILE="G_predictions.pkl"
IMG_EXTENSION="jpg"
CMAP="jet"

FIRST_YEAR=2001 # first year of input data, assuming first hour is 01/01/start_year
YEAR_START=2016
MONTH_START=1
DAY_START=1
YEAR_END=2016
MONTH_END=12
DAY_END=31

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
MAKE_PLOTS=true
LARGE_GRAPH=false

#--------------do not change here below------------------

if [ ${MAKE_PLOTS} = true ] ; then
	MAKE_PLOTS="--make_plots"
else
	MAKE_PLOTS="--no-make_plots"
fi

if [ ${LARGE_GRAPH} = true ] ; then
	LARGE_GRAPH="--large_graph"
else
	LARGE_GRAPH="--no-large_graph"
fi
