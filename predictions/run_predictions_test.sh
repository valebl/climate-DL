#!/bin/bash  
#SBATCH -A ict23_esp_0
#SBATCH --partition=boost_usr_prod
#SBATCH --time 24:00:00       # format: HH:MM:SS
#SBATCH -N 1                  # 1 node
#SBATCH --mem=60G
## SBATCH --ntasks-per-node=32   # 8 tasks out of 128
#SBATCH --gres=gpu:1          # 1 gpus per node out of 4
#SBATCH --job-name=pred_leo
## SBATCH --mail-type=FAIL,END
## SBATCH --mail-user=vblasone@ictp.it
#SBATCH -o /leonardo_work/ICT23_ESP_0/vblasone/climate-DL/predictions/north_italy/run.out
#SBATCH -e /leonardo_work/ICT23_ESP_0/vblasone/climate-DL/predictions/north_italy/run.err

source /leonardo/home/userexternal/vblasone/.bashrc

module purge
module load --auto profile/deeplrn
module load gcc
module load cuda/11.8

conda activate /leonardo_work/ICT23_ESP_0/env/GNNenv

cd /leonardo_work/ICT23_ESP_0/vblasone/climate-DL/predictions/

#"/leonardo_work/ICT23_ESP_0/vblasone/climate-DL/data_fvg_preprocessed/"

# north italy
python main.py --input_path="/leonardo_work/ICT23_ESP_0/SHARED/preprocessed/north_italy/" --output_path="/leonardo_work/ICT23_ESP_0/vblasone/climate-DL/predictions/north_italy/" --idx_file="idx_test.pkl" --graph_file_test="G_test.pkl" --subgraphs="subgraphs.pkl" --checkpoint_cl="/leonardo_work/ICT23_ESP_0/vblasone/climate-DL/local_single/cl_north_italy/checkpoint_4.pth" --checkpoint_reg="/leonardo_work/ICT23_ESP_0/vblasone/climate-DL/local_single/north_italy/checkpoint_24.pth" --input_file="input_standard.pkl" --lat_dim=15 --lon_dim=31 --model_name_cl="Classifier_edges_test" --model_name_reg="Regressor_edges_test" --output_file="G_predictions_2016_test.pkl" --log_file="log.txt" --make_plots --img_extension="jpg"

# fvg
#python main.py --input_path="/leonardo_work/ICT23_ESP_0/SHARED/preprocessed/fvg/" --output_path="/leonardo_work/ICT23_ESP_0/vblasone/climate-DL/predictions/test_leonardo/" --idx_file="idx_test.pkl" --graph_file_test="G_test.pkl" --subgraphs="subgraphs.pkl" --checkpoint_cl="/leonardo_work/ICT23_ESP_0/vblasone/climate-DL/local_single/test_cl_leo/checkpoint_20.pth" --checkpoint_reg="/leonardo_work/ICT23_ESP_0/vblasone/climate-DL/local_single/test_reg_leo/checkpoint_49.pth" --input_file="input_standard.pkl" --lat_dim=7 --lon_dim=7 --model_name_cl="Classifier_edges_test" --model_name_reg="Regressor_edges_test" --output_file="G_predictions_2016_test.pkl" --log_file="log.txt" --make_plots --img_extension="jpg"

#python main.py --input_path="/m100_work/ICT23_ESP_C/vblasone/climate-DL/data_north_preprocessed/" --output_path="/m100_work/ICT23_ESP_C/vblasone/climate-DL/predictions/test_north-from-fvg_2016_cl_e24_reg-e49_z_only/" --idx_file="idx_test.pkl" --graph_file_test="G_test.pkl" --subgraphs="subgraphs.pkl" --checkpoint_cl="/m100_work/ICT23_ESP_C/vblasone/climate-DL/local_single/cl_test_z_only/checkpoint_24.pth" --checkpoint_reg="/m100_work/ICT23_ESP_C/vblasone/climate-DL/local_single/reg_test_z_only/checkpoint_49.pth" --input_file="input_standard.pkl" --lat_dim=15 --lon_dim=31 --model_name_cl="Classifier_z_only_test" --model_name_reg="Regressor_z_only_test" --output_file="G_predictions_2016_test.pkl" --log_file="log.txt" --make_plots --img_extension="jpg"


