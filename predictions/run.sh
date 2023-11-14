#!/bin/bash
source $1
mkdir -p ${LOG_PATH}

sbatch << EOT
#!/bin/bash
#SBATCH -A ict23_esp_0
#SBATCH -p boost_usr_prod
#SBATCH --time=${TIME}       # format: HH:MM:SS
#SBATCH -N 1                  # 1 node
#SBATCH --mem=${MEM}
#SBATCH --ntasks-per-node=32 # out of 128
#SBATCH --gres=gpu:4          # 1 gpus per node out of 4
#SBATCH --job-name=${JOB_NAME}
#SBATCH --mail-type=FAIL,END
# SBATCH --mail-user=${MAIL}
#SBATCH -o ${LOG_PATH}/run.out
#SBATCH -e ${LOG_PATH}/run.err

module purge
module load --auto profile/deeplrn
module load gcc
module load cuda/11.8 

source ${SOURCE_PATH}

conda activate ${ENV_PATH}

cd ${MAIN_PATH}

## predictions
python main.py --input_path=${INPUT_PATH} --output_path=${OUTPUT_PATH} --input_file=${INPUT_FILE} --test_graph_file=${TEST_GRAPH_FILE} --subgraphs_file=${SUBGRAPHS_FILE} --checkpoint_cl=${CHECKPOINT_CL} --checkpoint_reg=${CHECKPOINT_REG} --model_name_cl=${MODEL_NAME_CL} --model_name_reg=${MODEL_NAME_REG} --output_file=${OUTPUT_FILE} --log_file=${LOG_FILE} --lon_min=${LON_MIN} --lon_max=${LON_MAX} --lat_min=${LAT_MIN} --lat_max=${LAT_MAX} ${MAKE_PLOTS} ${LARGE_GRAPH} --img_extension=${IMG_EXTENSION} --year_start=${YEAR_START} --month_start=${MONTH_START} --day_start=${DAY_START} --year_end=${YEAR_END} --month_end=${MONTH_END} --day_end=${DAY_END} --first_year=${FIRST_YEAR} --cmap=${CMAP} --batch_size=${BATCH_SIZE}
EOT

