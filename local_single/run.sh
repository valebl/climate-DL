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

if [ -z ${WEIGHTS_FILE} ]; then
	WEIGHTS=""
else
	WEIGHTS="--weights_file=${WEIGHTS_FILE}"
fi

## training
accelerate launch --config_file ${ACCELERATE_CONFIG_PATH} main.py --input_path=${INPUT_PATH} --output_path=${OUTPUT_PATH} --input_file=${INPUT_FILE} --idx_file=${IDX_FILE} --log_file=${LOG_FILE} --target_file=${TARGET_FILE} --mask_target_file=${MASK_TARGET_FILE} --graph_file=${GRAPH_FILE} --subgraphs_file=${SUBGRAPHS_FILE} --out_checkpoint_file=${OUT_CHECKPOINT_FILE} --out_loss_file=${OUT_LOSS_FILE} --pct_trainset=${PCT_TRAINING} --epochs=${EPOCHS} --batch_size=${BATCH_SIZE} --step_size=${LR_STEP_SIZE} --lr=${LR} --weight_decay=${WEIGHT_DECAY} --model_name=${MODEL_NAME} --loss_fn=${LOSS_FN} --model_type=${MODEL_TYPE} --use_accelerate --load_checkpoint --no-test_model --no-ctd_training --fine_tuning --checkpoint_file=${CHECKPOINT_FILE} --wandb_project_name=${WANDB_PROJECT_NAME} --lon_min=${LON_MIN} --lon_max=${LON_MAX} --lat_min=${LAT_MIN} --lat_max=${LAT_MAX} ${USE_ACCELERATE} ${LOAD_CHECKPOINT} ${CTD_TRAINING} ${FINE_TUNING} \${WEIGHTS}
EOT

