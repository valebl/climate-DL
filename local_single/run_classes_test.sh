#!/bin/bash

for /f "delims=" %%x in (config.txt) do (set "%%x")

#SBATCH -A ict23_esp_0
#SBATCH -p boost_usr_prod
#SBATCH --time 01:00:00       # format: HH:MM:SS
#SBATCH -N 1                  # 1 node
#SBATCH --mem=60G
#SBATCH --ntasks-per-node=32 # out of 128
#SBATCH --gres=gpu:4          # 1 gpus per node out of 4
#SBATCH --job-name=cl_test
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=vblasone@ictp.it
#SBATCH -o /leonardo_work/ICT23_ESP_0/vblasone/climate-DL/local_single/cl_test/run.out
#SBATCH -e /leonardo_work/ICT23_ESP_0/vblasone/climate-DL/local_single/cl_test/run.err

module purge
module load --auto profile/deeplrn
module load gcc
module load cuda/11.8 

source /leonardo/home/userexternal/vblasone/.bashrc

conda activate /leonardo_work/ICT23_ESP_0/env/GNNenv

cd /leonardo_work/ICT23_ESP_0/vblasone/climate-DL/local_single/

## training
accelerate launch --config_file /leonardo/home/userexternal/vblasone/.cache/huggingface/accelerate/default_config_4.yaml main.py --input_path="/leonardo_work/ICT23_ESP_0/vblasone/climate-DL/data_fvg_preprocessed/" --output_path="/leonardo_work/ICT23_ESP_0/vblasone/climate-DL/local_single/cl_test/" --input_file="input_standard.pkl"  --idx_file="idx_train_cl.pkl" --log_file="log.txt" --target_file="target_train_cl.pkl" --mask_target_file="mask_train_cl.pkl" --graph_file="G_train.pkl" --mask_target_file="mask_train_cl.pkl" --subgraphs_file="subgraphs.pkl" --out_checkpoint_file="checkpoint.pth" --out_loss_file="loss.csv" --pct_trainset=1 --epochs=25 --batch_size=64 --step_size=25 --lr=0.0001 --weight_decay=0.0 --model_name="Classifier_old" --loss_fn="sigmoid_focal_loss" --model_type="cl" --use_accelerate --load_checkpoint --no-test_model --no-ctd_training --fine_tuning --performance="accuracy"  --checkpoint_file="/leonardo_work/ICT23_ESP_0/vblasone/climate-DL/data_fvg_preprocessed/checkpoint_ae_e3.pth" --wandb_project_name="Classifier-test-Leonardo" --mode="train" --lon_min=12.75 --lon_max=14.00 --lat_min=45.50 --lat_max=46.75

## continue the training
#accelerate launch --config_file ~/.cache/huggingface/accelerate/default_config_4.yaml main.py --input_path="/m100_work/ICT23_ESP_C/vblasone/climate-DL/data_fvg_preprocessed/" --output_path="/m100_work/ICT23_ESP_C/vblasone/climate-DL/local_single/cl_test/ctd/" --input_file="input_standard.pkl"  --idx_file="idx_train_cl.pkl" --log_file="log.txt" --target_file="target_train_cl.pkl" --mask_target_file="mask_train_cl.pkl" --graph_file="G_train.pkl" --mask_target_file="mask_train_cl.pkl" --subgraphs_file="subgraphs.pkl" --out_checkpoint_file="checkpoint.pth" --out_loss_file="loss.csv" --pct_trainset=1 --epochs=25 --batch_size=64 --step_size=25 --lr=0.0001 --weight_decay=0.0 --model_name="Classifier_old" --loss_fn="sigmoid_focal_loss" --model_type="cl" --use_accelerate --no-load_checkpoint --no-test_model --ctd_training --fine_tuning --performance="accuracy"  --checkpoint_ctd="/m100_work/ICT23_ESP_C/vblasone/climate-DL/local_single/cl_test_wd0/checkpoint_24.pth" --wandb_project_name="Classifier-test-ctd" --mode="train" --lon_min=12.75 --lon_max=14.00 --lat_min=45.50 --lat_max=46.75

