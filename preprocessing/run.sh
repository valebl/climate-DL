#!/bin/bash
source $1
mkdir -p ${OUTPUT_PATH}

sbatch << EOT
#!/bin/bash
#SBATCH -A ict23_esp_0
#SBATCH -p boost_usr_prod
#SBATCH --time ${TIME}       # format: HH:MM:SS
#SBATCH -N 1                  # 1 node
#SBATCH --mem=${MEM}
#SBATCH --ntasks-per-node=32   # 8 tasks out of 128
#SBATCH --job-name=${JOB_NAME}
# SBATCH --mail-type=FAIL,END
# SBATCH --mail-user=${MAIL}
#SBATCH -o ${LOG_PATH}/run.out
#SBATCH -e ${LOG_PATH}/run.err

#----------#
# PHASE 1A #
#----------#
module purge
module load --auto profile/meteo
module load cdo

source ${SOURCE_PATH}

cd ${INPUT_PATH_PHASE_1A}

if [ ${PERFORM_PHASE_1A} = true ] ; then
	source ${PHASE_1A_PATH} ${LON_MIN} ${LON_MAX} ${LAT_MIN} ${LAT_MAX} ${INTERVAL} ${INPUT_PATH_PHASE_1A} ${OUTPUT_PATH} ${PREFIX_PHASE_1}
fi

#----------#
# PHASE 1B #
#----------#
module purge
module load --auto profile/deeplrn
module load gcc
module load cuda/11.8
module load hdf5/1.12.2--gcc--11.3.0
module load netcdf-c/4.9.0--gcc--11.3.0

conda activate ${ENV_PATH}

if [ ${PERFORM_PHASE_1B} = true ] ; then
	python3 ${PHASE_1B_PATH} --input_path=${INPUT_PATH_PHASE_1B} --output_path=${OUTPUT_PATH} --input_files_prefix=${PREFIX_PHASE_1} --stats_path=${STATS_PATH} --means_file=${MEANS_FILE} --stds_file=${STDS_FILE} ${LOAD_STATS}
fi

#---------#
# PHASE 2 #
#---------#

if [ ${PERFORM_PHASE_2} = true ] ; then
	python3 ${PHASE_2_PATH} --input_path=${INPUT_PATH_PHASE_2} --output_path=${OUTPUT_PATH} --log_file=${LOG_FILE} --lon_min=${LON_MIN} --lon_max=${LON_MAX} --lat_min=${LAT_MIN} --lat_max=${LAT_MAX} --interval=${INTERVAL} --suffix=${SUFFIX_PHASE_2} --idx_time_path=${IDX_TIME_PATH} --make_plots --precomputed_stats_file=${PRECOMPUTED_STATS_FILE} --use_precomputed_stats
fi
EOT

