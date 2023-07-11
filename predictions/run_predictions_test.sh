#!/bin/bash  
#SBATCH -A ict23_esp_C
#SBATCH --partition=m100_usr_prod
## SBATCH --qos=qos_prio
#SBATCH --time 00:20:00       # format: HH:MM:SS
#SBATCH -N 1                  # 1 node
#SBATCH --mem=60G
## SBATCH --ntasks-per-node=32   # 8 tasks out of 128
#SBATCH --gres=gpu:1          # 1 gpus per node out of 4
#SBATCH --job-name=pred-09-5
## SBATCH --mail-type=FAIL,END
## SBATCH --mail-user=vblasone@ictp.it
#SBATCH -o /m100_work/ICT23_ESP_C/vblasone/climate-DL/predictions/focal_loss_tuning/alpha_09_gamma_5/run.out
#SBATCH -e /m100_work/ICT23_ESP_C/vblasone/climate-DL/predictions/focal_loss_tuning/alpha_09_gamma_5/run.err

source /m100/home/userexternal/vblasone/.bashrc

conda activate /m100/home/userexternal/vblasone/geometric

cd /m100_work/ICT23_ESP_C/vblasone/climate-DL/predictions/

python main.py --input_path="/m100_work/ICT23_ESP_C/vblasone/climate-DL/data_fvg_preprocessed/" --output_path="/m100_work/ICT23_ESP_C/vblasone/climate-DL/predictions/focal_loss_tuning/alpha_09_gamma_5/" --idx_file="idx_test.pkl" --graph_file_test="G_test.pkl" --subgraphs="subgraphs.pkl" --checkpoint_cl="/m100_work/ICT23_ESP_C/vblasone/climate-DL/local_single/focal_loss_tuning/cl/alpha_09_gamma_5/checkpoint_24.pth" --checkpoint_reg="/m100_work/ICT23_ESP_C/vblasone/climate-DL/local_single/reg_test_edges/checkpoint_49.pth" --input_file="input_standard.pkl" --lat_dim=7 --lon_dim=7 --model_name_cl="Classifier_edges_test" --model_name_reg="Regressor_edges_test" --output_file="G_predictions_2016_test.pkl" --log_file="log.txt" --make_plots --img_extension="jpg"

#python main.py --input_path="/m100_work/ICT23_ESP_C/vblasone/climate-DL/data_north_preprocessed/" --output_path="/m100_work/ICT23_ESP_C/vblasone/climate-DL/predictions/test_north-from-fvg_2016_cl_e24_reg-e49_z_only/" --idx_file="idx_test.pkl" --graph_file_test="G_test.pkl" --subgraphs="subgraphs.pkl" --checkpoint_cl="/m100_work/ICT23_ESP_C/vblasone/climate-DL/local_single/cl_test_z_only/checkpoint_24.pth" --checkpoint_reg="/m100_work/ICT23_ESP_C/vblasone/climate-DL/local_single/reg_test_z_only/checkpoint_49.pth" --input_file="input_standard.pkl" --lat_dim=15 --lon_dim=31 --model_name_cl="Classifier_z_only_test" --model_name_reg="Regressor_z_only_test" --output_file="G_predictions_2016_test.pkl" --log_file="log.txt" --make_plots --img_extension="jpg"


