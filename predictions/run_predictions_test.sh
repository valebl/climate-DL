#!/bin/bash  
#SBATCH -A ict23_esp_C
#SBATCH --partition=m100_usr_prod
## SBATCH --qos=qos_prio
#SBATCH --time 00:15:00       # format: HH:MM:SS
#SBATCH -N 1                  # 1 node
#SBATCH --mem=60G
## SBATCH --ntasks-per-node=32   # 8 tasks out of 128
#SBATCH --gres=gpu:1          # 1 gpus per node out of 4
#SBATCH --job-name=pred
## SBATCH --mail-type=FAIL,END
## SBATCH --mail-user=vblasone@ictp.it
#SBATCH -o /m100_work/ICT23_ESP_C/vblasone/climate-DL/predictions/run.out
#SBATCH -e /m100_work/ICT23_ESP_C/vblasone/climate-DL/predictions/run.err

source /m100/home/userexternal/vblasone/.bashrc

conda activate /m100/home/userexternal/vblasone/geometric

cd /m100_work/ICT23_ESP_C/vblasone/climate-DL/predictions/

python main.py --input_path="/m100_work/ICT23_ESP_C/vblasone/climate-DL/data_fvg_preprocessed/" --output_path="/m100_work/ICT23_ESP_C/vblasone/climate-DL/predictions/test_fvg_2016/" --idx_file="idx_test.pkl" --graph_file_test="G_test.pkl" --subgraphs="subgraphs.pkl" --checkpoint_cl="/m100_work/ICT23_ESP_C/vblasone/climate-DL/local_single/cl_test_wd0/ctd/checkpoint_44.pth" --checkpoint_reg="/m100_work/ICT23_ESP_C/vblasone/climate-DL/local_single/reg_test_wd0/ctd/checkpoint_99.pth" --input_file="input_standard.pkl" --lat_dim=7 --lon_dim=7 --model_name_cl="Classifier_old_test" --model_name_reg="Regressor_old_test" --output_file="G_predictions_2016_test_wd0.pkl" --log_file="log.txt" --make_plots


